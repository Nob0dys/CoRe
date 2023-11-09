import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
## for complexity
# from thop import profile, clever_format
# from ptflops import get_model_complexity_info
import ipdb
import sys
import itertools
from utils.parser import extract_weight_method_parameters_from_args
from methods.weight_methods import WeightMethods
from timm.scheduler import CosineLRScheduler
# sys.path.append(".")
sys.path.append("..")
from models.CoRe import CoRe

train_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict




def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]




def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)

    # build model
    # base_model = builder.model_builder(config.model, args)
    base_model = CoRe(config.model, args).cuda()
    if args.use_gpu:
        base_model.to(args.local_rank)

 
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    print(args.start_ckpts)
    # resume ckpts
    if args.resume:
        print('start the resume process')
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)


    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
            # Change find_unused_parameters=False
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=False)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()

   
    # optimizer & scheduler
    optimizer, scheduler_default = builder.build_opti_sche(base_model, config)
    

    # trainval
    # training
    base_model.zero_grad()

    reset_optimizer = True

    # build the queue
    queue = None
    queue_path = os.path.join(args.experiment_path, 'queue' + str(args.local_rank) + '.pth')

   

    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)['queue']
 

    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (config.total_bs * args.world_size)  # 8192

    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()
    

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_xyz = AverageMeter(['Loss'])
        losses_normal = AverageMeter(['Loss'])
        losses_MP = AverageMeter(['Loss'])

        num_iter = 0

        gradual_weight = float(epoch) / float(config.max_epoch)
      
        
        n_batches = len(train_dataloader)

        # optionally starts a queue
        if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
            queue = torch.zeros(
                np.sum(args.nmb_crops), 
                args.queue_length // args.world_size, 
                args.output_dim
                ).cuda()
        
        use_the_queue = False

        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)   
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            points = train_transforms(points)  # [128, 1024, 6]
          
            loss_xyz, loss_normal, loss_MP, queue, representation = base_model(points, queue, use_the_queue)
            
           
            ###################### calculating complexity
            # macs, params = get_model_complexity_info(base_model, (2, 1024, 6), as_strings=False,
            #                                          print_per_layer_stat=True, verbose=False)
            # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            # ipdb.set_trace()

            if config.loss_type == 'xyz':
                loss = loss_xyz
            elif config.loss_type == 'normal':
                loss = float(config.normal_weight) * loss_normal
            elif config.loss_type == 'xyznormal':
                loss = loss_xyz + float(config.normal_weight) * loss_normal
            elif config.loss_type == 'xyznormal_gradual':
                loss = loss_xyz + gradual_weight * float(config.normal_weight) * (loss_normal + loss_MP)
                
            elif config.loss_type == 'automatic_weight':
                loss = awl(loss_xyz, loss_normal, loss_MP)
                
            elif config.loss_type == 'xyznormal_xyzfirst':
                if epoch < 300:
                    loss = loss_xyz
                else:
                    loss = loss_xyz + float(config.normal_weight) * loss_normal
            elif config.loss_type == 'xyznormal_xyzfirst_gradual':
                if epoch < 300:
                    loss = loss_xyz
                else:
                    if reset_optimizer:
                        reset_optimizer = False
                        optimizer, scheduler_default = builder.build_opti_sche(base_model, config)
                    gradual_weight = float(epoch-299) / float(config.max_epoch-299)
                    loss = loss_xyz + float(config.normal_weight) * loss_normal * gradual_weight
            else:
                raise NotImplementedError

            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses_xyz.update([loss_xyz.item() * 1000])
                losses_normal.update([loss_normal.item() * 1000])
                losses_MP.update([loss_MP.item() * 1000])
            else:
                losses_xyz.update([loss_xyz.item() * 1000])
                losses_normal.update([loss_normal.item() * 1000])
                losses_MP.update([loss_MP.item() * 1000])
                # RuntimeError: a Tensor with 2 elements cannot be converted to Scalar
                # Solution: use torch.mean() to convert it to a one element tensor
               


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)
                


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss_xyz = %s Loss_normal = %s Loss_MP = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses_xyz.val()], ['%.4f' % l for l in losses_normal.val()], ['%.4f' % l for l in losses_MP.val()], optimizer.param_groups[0]['lr']), logger = logger)
                
        if isinstance(scheduler_default, list):
            for item in scheduler_default:
                item.step(epoch)
        else:
            scheduler_default.step(epoch)

        # if isinstance(scheduler, list):
        #     for item in scheduler:
        #         item.step(epoch)
        # else:
        #     scheduler.step(epoch)



        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/LR', optimizer.param_groups[0]['lr'], epoch)
            train_writer.add_scalar('Loss/Epoch/Loss', loss.item(), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_xyz', losses_xyz.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_normal', losses_normal.avg(0), epoch)
            train_writer.add_scalar('Loss/Epoch/Loss_MP', losses_MP.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Loss_xyz = %s Loss_normal = %s Loss_MP = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses_xyz.avg()], ['%.4f' % l for l in losses_normal.avg()], ['%.4f' % l for l in losses_MP.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)


       

        # if epoch % args.val_freq == 0 and epoch != 0:
        #     # Validate the current model
        #     metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)
        #
        #     # Save ckeckpoints
        #     if metrics.better_than(best_metrics):
        #         best_metrics = metrics
        #         builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 25 ==0 and epoch >=250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)

        # save the queue
        if queue is not None:
        # if type(queue) == torch.Tensor:
            # if not os.path.exists(args.exp_name):
            #     os.makedirs(args.exp_name)
            # print('exit')
            # sys.exit()
            torch.save({'queue': queue}, queue_path)

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, extra_train_dataloader, test_dataloader, epoch, queue, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():

        use_the_queue_extra_train = False
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, queue, use_the_queue_extra_train, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        use_the_queue_test = False
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, queue, use_the_queue_test, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())


        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass
