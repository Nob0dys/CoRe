import os
import argparse
from pathlib import Path
from collections import defaultdict
import sys
sys.path.append("..")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        default='cfgs/pretrain_CoRe.yaml',
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local_rank', type=int, default=0, help="this argument is not used and should be ignored")
    parser.add_argument('--num_workers', type=int, default=8)
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    # when args.distributed=True, set sync_bn=True
    parser.add_argument(
        '--sync_bn', 
        action='store_false', 
        default=True, 
        help='whether to use sync bn')
    

 


    # queue and multual prediction
    parser.add_argument('--large_group', type = int, default=5, help = 'number of large groups')
    parser.add_argument('--small_group', type = int, default=10, help = 'number of small groups')
    parser.add_argument('--large_patch', type = int, default=6, help = 'number of large patches')
    parser.add_argument('--small_patch', type = int, default=3, help = 'number of small patches')
    parser.add_argument('--queue_length', type = int, default=8192, help = 'queue length')
    parser.add_argument('--epoch_queue_starts', type = int, default=10, help = 'from this epoch, we start using a queue')
    parser.add_argument("--nmb_crops", type=int, default=26, nargs="+", help="list of number of crops")  # 26
    parser.add_argument('--fea_dim', type = int, default=384, help = 'feature dimension of MaskEncoder')
    parser.add_argument('--hidden_mlp', type = int, default=256, help = 'hidden dimension of projection head')
    parser.add_argument('--output_dim', type = int, default=128, help = 'output dimension of projection head')
    parser.add_argument('--nmb_prototypes', type = int, default=200, help = 'number of prototypes')  # 200
    parser.add_argument('--temperature', type = float, default=0.1, help = 'temperature to control MP_loss')
    parser.add_argument('--epsilon', type = float, default=0.05, help = 'regularization parameter for Sinkhorn-Knopp algorithm')
    parser.add_argument('--sinkhorn_iterations', type = int, default=3, help = 'number of iterations in Sinkhorn-Knopp algorithm')
    parser.add_argument("--world_size", default=1, type=int, help="number of processes: it is set automatically and should not be passed as argument")
   

    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help = 'vote acc')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model', 
        action='store_true', 
        default=False, 
        help = 'finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model', 
        action='store_true', 
        default=False, 
        help = 'training modelnet from scratch')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')        
    parser.add_argument(
        '--way', type=int, default=-1)
    parser.add_argument(
        '--shot', type=int, default=-1)
    parser.add_argument(
        '--fold', type=int, default=-1)
    
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        print(
            'training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode
    
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)


def extract_weight_method_parameters_from_args(args):
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=args.update_weights_every,
                optim_niter=args.nashmtl_optim_niter,
            ),
            stl=dict(main_task=args.main_task),
            cagrad=dict(c=args.c),
            dwa=dict(temp=args.dwa_temp),
        )
    )
    return weight_methods_parameters


