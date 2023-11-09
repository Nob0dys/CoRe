import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, ChamferDistanceL2_withnormal, \
    ChamferDistanceL2_withnormal_strict, ChamferDistanceL2_withnormal_strict_normalindex, ChamferDistanceL2_withnormal_normalindex, ChamferDistanceL2_withnormalL1
import sys
import os

        


class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3 or B N 6
            ---------------------------
            output: B G M 3 or B G M 6
            center : B G 3 or B G 6
        '''
        # for complexity calculation.
        # print(xyz.shape)
        # if xyz.shape[0] == 1:
        #     xyz = xyz[0]
        batch_size, num_points, _ = xyz.shape
        xyz_no_normal = xyz[:, :, :3].clone().contiguous()  # former three is positions
        xyz_only_normal = xyz[:, :, 3:6].clone().contiguous()  # later three is normals
        # fps the centers out
        _, center = misc.fps(xyz_no_normal, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz_no_normal, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        # neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        # neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 6).contiguous()

        neighborhood_no_normal = xyz_no_normal.view(batch_size * num_points, -1)[idx, :]
        neighborhood_no_normal = neighborhood_no_normal.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood_no_normal = neighborhood_no_normal - center.unsqueeze(2)

        neighborhood_only_normal = xyz_only_normal.view(batch_size * num_points, -1)[idx, :]
        neighborhood_only_normal = neighborhood_only_normal.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # neighborhood_no_normal[:, :, :, :3] = neighborhood[:, :, :, :3].contiguous() - center.unsqueeze(2)[:, :, :, :3].contiguous()
        return neighborhood_no_normal, neighborhood_only_normal, center


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)

        x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool() 

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        
        group_input_tokens = self.encoder(neighborhood)  #  B G C [128, 64, 384]

        batch_size, seq_len, C = group_input_tokens.size()

      
        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)  # [128, 26, 384]
        # add pos embedding
        # visble and mask pos center
        vis_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        mask_center = center[bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(vis_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis, bool_masked_pos, vis_center, mask_center



# Mutual prediction module
class Mutual_Prediction(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fea_dim = args.fea_dim
        self.hidden_mlp = args.hidden_mlp
        self.output_dim = args.output_dim
        self.nmb_prototypes = args.nmb_prototypes
        self.MP_patch_num = args.nmb_crops
        # self.MP_patch_num = args.sample_num
        self.temperature = args.temperature
        self.epsilon = args.epsilon
        self.sinkhorn_iterations = args.sinkhorn_iterations
        self.world_size = args.world_size
        self.distributed = args.distributed
    
        self.projection_head = nn.Sequential(
                nn.Linear(self.fea_dim, self.hidden_mlp),
                # nn.BatchNorm1d(self.MP_patch_num),  # hidden_mlp -> MP_patch_num
                nn.BatchNorm1d(self.hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_mlp, self.output_dim))
    
        self.prototype_head = nn.Linear(self.output_dim, self.nmb_prototypes, bias=False)

       

        self.apply(self._init_weights)

    # TODO: check the init function
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)



    def _forward_head(self, x_vis_2d):
        # x_vis_2d.size() = [bs * MP_patch_num, fea_dim]
        # x_vis.size() = [bs, MP_patch_num, fea_dim]
        x_vis_2d = self.projection_head(x_vis_2d)  # x_vis_2d.size() = [bs * MP_patch_num, output_dim]
        x_vis_2d =  nn.functional.normalize(x_vis_2d, dim=1, p=2)  # dim=1 -> dim=2

        return x_vis_2d, self.prototype_head(x_vis_2d)  
        # [bs, MP_patch_num, output_dim] & [bs, MP_patch_num, nmb_prototypes]

    
    def forward(self, x_vis, queue, use_the_queue):
        with torch.no_grad():
            w = self.prototype_head .weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototype_head.weight.copy_(w)

        # make x_vis from 3D tensor to 2D
        # [bs, MP_patch_num, fea_dim] -> [bs * MP_patch_num, fea_dim]
        bs = x_vis.size(0)
        x_vis_2d = x_vis.view(-1, self.fea_dim)  # x_vis_2d.size() = [bs * MP_patch_num, fea_dim]


        # embedding.size() = [bs * MP_patch_num, output_dim]
        # output.size() = [bs * MP_patch_num, nmb_prototypes]
        embedding, output = self._forward_head(x_vis_2d)  
        
        # output = output.view(-1, self.nmb_prototypes)  # output.size() = [bs * MP_patch_num, nmb_prototypes]
        # embedding = output.view(-1, self.output_dim)  # embedding.size() = [bs * MP_patch_num, output_dim]
        embedding = embedding.detach()  
        

        # ============ multual prediction loss ... ============
        MP_loss = 0
        for i, crop_id in enumerate(np.arange(self.MP_patch_num)):
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)].detach()  # out.size = [bs, nmb_prototypes]

                # time to use the queue
                if queue is not None:
              
                    if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                        use_the_queue = True
    
                        out = torch.cat((torch.mm(
                                queue[i],
                                self.prototype_head.weight.t()
                            ), out))
                      
                    queue[i, bs:] = queue[i, :-bs].clone()
                    queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                q = self._distributed_sinkhorn(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(self.MP_patch_num), crop_id):
                x = output[bs * v: bs * (v + 1)] / self.temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            MP_loss += subloss / (self.MP_patch_num - 1)
        MP_loss /= self.MP_patch_num


        return MP_loss, queue
    
    @torch.no_grad()
    def _distributed_sinkhorn(self, out):
        # -------------------------------------- #
        # 将B个samples分配到K个prototypes中的最优分配
        # -------------------------------------- #
        Q = torch.exp(out / self.epsilon).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * self.world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if self.distributed == True:
            torch.distributed.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if self.distributed == True:
                torch.distributed.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    






@MODELS.register_module()
class CoRe(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        print_log(f'[CoRe] ', logger ='CoRe')
        self.config = config
        self.args = args
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.MP_module = Mutual_Prediction(args)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        

        print_log(f'[CoRe] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='CoRe')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
       

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        self.increase_dim2 = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

  
        
        
    
    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type =='cdl2normal':
            self.loss_func = ChamferDistanceL2_withnormal().cuda()
        # elif loss_type == 'cdl2normall1':
        #     self.loss_func = ChamferDistanceL2_withnormalL1().cuda()
        # elif loss_type == 'cdl2normal_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_normalindex().cuda()
        # elif loss_type =='cdl2normalstrict':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict().cuda()
        # elif loss_type == 'cdl2normalstrict_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict_normalindex().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def forward(self, pts, queue, use_the_queue, vis=False, vis_surfel=False, **kwargs):
      


        neighborhood, neighborhood_normal, center = self.group_divider(pts)
        # neighborhood.size = [128, 64, 32, 3]
        # neighborhood_normal.size = [128, 64, 32, 3]
        # center.size = [128, 64, 3]
        

        x_vis, mask, vis_center, mask_center = self.MAE_encoder(neighborhood, center)
        B, vis_num, C = x_vis.shape # B VIS C
        
        # Multual prediction applies to the visible multi-scale patches
        MP_loss, queue = self.MP_module(x_vis, queue, use_the_queue)
        

        pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)  # [128, 26, 384]

        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)  # [128, 38, 384]
        
        _,N,_ = pos_emd_mask.shape  # N = 38
        mask_token = self.mask_token.expand(B, N, -1)  # 维度为[128, 38, 384]的零矩阵
        x_full = torch.cat([x_vis, mask_token], dim=1)  # [128, 26 + 38, 384]
        pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)  # [128, 26 + 38, 384]

        x_rec = self.MAE_decoder(x_full, pos_full, N)
        B, M, C = x_rec.shape  # B = 128, M = 38, C = 384
      
      
        rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 3 [4864, 32, 3]
        rebuild_normal = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 3 [4864, 32, 3]
    

        gt_points = neighborhood[mask].reshape(B * M, -1, 3) ## BM, Gs, 3
        gt_normals = neighborhood_normal[mask].reshape(B * M, -1, 3)  ## BM, Gs, 3
        loss_xyz, loss_normal = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals)



        if vis: #visualization
          
            # --------------
            # point position 
            # --------------
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)  # [26, 32, 3], visible patches without centers
            full_vis = vis_points + center[~mask].unsqueeze(1)  # [26, 32, 3], visible patches with centers
            full_rebuild = rebuild_points + center[mask].unsqueeze(1)  # [38, 32, 3], rebuild patches with centers
            full = torch.cat([full_vis, full_rebuild], dim=0)  # [64, 32, 3], the completed point 

            # full_points = torch.cat([rebuild_points,vis_points], dim=0)  # [64, 32, 3]
            full_center = torch.cat([center[mask], center[~mask]], dim=0)  # [64, 3]
            # full = full_points + full_center.unsqueeze(1)  # [64, 32, 3]

            full_position = full.reshape(-1, 3).unsqueeze(0)  # [1, 64 * 32, 3]
            vis_position = full_vis.reshape(-1, 3).unsqueeze(0) # [1, 26 * 32, 3]
            

            # --------------
            # point normals 
            # --------------
            vis_normal = neighborhood_normal[~mask].reshape(B * (self.num_group - M), -1, 3)  # [26, 32, 3]
            # vis_normal = vis_normal + center[~mask].unsqueeze(1)  # [26, 32, 3], visible normal with centers
            # rebuild_normal = rebuild_normal + center[mask].unsqueeze(1)  # [38, 32, 3], rebuild normal with centers # [38, 32, 3]
            full_normal = torch.cat([vis_normal, rebuild_normal], dim=0)  # [64, 32, 3]
            full_normal = full_normal.reshape(-1, 3).unsqueeze(0)  # [1, 64 * 32, 3]
            vis_normal = vis_normal.reshape(-1, 3).unsqueeze(0) # [1, 26 * 32, 3]


            if vis_surfel:
                input_vis_point = vis_position.squeeze()  # [26 * 32, 3]
                output_surfels = torch.zeros(38 * 32, 8)
                output_surfels[:, :3] = full_rebuild.reshape(-1, 3)
                output_surfels[:, 3:6] = rebuild_normal.reshape(-1, 3)
                output_surfels[:, 6] = 0
                output_surfels[:, 7] = 0

                return input_vis_point, output_surfels

            else:
                return full_position, vis_position, full_center
        else:
            return loss_xyz, loss_normal, MP_loss, queue, x_vis

       
