import sys, os

import blobfile as bf
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from torchvision.models import *
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR, StepLR

sys.path.append('.')
from improved_diffusion import dist_util

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def reward_function(delta_err, alpha=1.0):
    cur_reward = max(np.exp(alpha*delta_err)-1, 0)
    
    return cur_reward

def create_model(model_type='efficientnet', in_dim=1, out_dim=1):
    vf_predictor = None
    if model_type == 'vit':
        vf_predictor = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vf_predictor.conv_proj = nn.Conv2d(in_dim, 768, kernel_size=(16, 16), stride=(16, 16))
        vf_predictor.heads[0] = nn.Linear(in_features=768, out_features=out_dim, bias=True)
    elif model_type == 'efficientnet':
        vf_predictor = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        vf_predictor.features[0][0] = nn.Conv2d(in_dim, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        vf_predictor.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)
    elif model_type == 'resnet':
        vf_predictor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        vf_predictor.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'swin':
        vf_predictor = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        vf_predictor.features[0][0] = nn.Conv2d(in_dim, 128, kernel_size=(4, 4), stride=(4, 4))
        vf_predictor.head = nn.Linear(in_features=1024, out_features=out_dim, bias=True)
    elif model_type == 'vgg':
        vf_predictor = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        vf_predictor.features[0] = nn.Conv2d(in_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        vf_predictor.classifier[6] = nn.Linear(in_features=4096, out_features=out_dim, bias=True)
    elif model_type == 'resnext':
        vf_predictor = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
        vf_predictor.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'wideresnet':
        vf_predictor = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        vf_predictor.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'convnext':
        vf_predictor = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        vf_predictor.features[0][0] = nn.Conv2d(in_dim, 96, kernel_size=(4, 4), stride=(4, 4))
        vf_predictor.classifier[2] = nn.Linear(in_features=768, out_features=out_dim, bias=True)
    return vf_predictor

class Model_Wrapper():
    """docstring for Model_Wrapper"""
    def __init__(self, model,
                result_dir='.',
                resume_checkpoint=None,
                ema_rate=0.9,
                identifier='predictor',
                logger=None):
        super().__init__()
        self.model = model
        self.model_params = list(model.parameters())
        # self.model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        self.ddp_model = DDP(
                model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        self.identifier = identifier

        self.resume_checkpoint = resume_checkpoint
        self.resume_epoch = 0
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )

        self.result_dir = result_dir
        self.logger = logger 

    def _log(self, text):
        if self.logger is not None:
            self.logger.log(text)
        else:
            print(text)

    def _load_checkpoint(self, opt=None):
        self._load_and_sync_parameters()
        if self.resume_checkpoint:
            if opt is not None:
                opt = self._load_optimizer_state(opt)
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.model_params) for _ in range(len(self.ema_rate))
            ]
        return opt

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_epoch = parse_resume_step_from_filename_(resume_checkpoint)
            if dist.get_rank() == 0:
                self._log(f"{self.identifier} - loading vf predictor from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.model_params)

        # main_checkpoint = resume_checkpoint
        ema_checkpoint = find_ema_checkpoint_(self.resume_checkpoint, self.resume_epoch, rate, self.identifier)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                self._log(f"{self.identifier} - loading vf EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _state_dict_to_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        # if self.use_fp16:
        #     return make_master_params(params)
        # else:
        return params

    def save(self, epoch, opt):
        def save_checkpoint(rate, epoch, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                self._log(f"saving {self.identifier} {rate}...")
                if not rate:
                    filename = f"{self.identifier}_{epoch:06d}.pt"
                else:
                    filename = f"ema_{self.identifier}_{rate}_{epoch:06d}.pt"
                with bf.BlobFile(bf.join(self.result_dir, filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint(0, epoch, self.model_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, epoch, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.result_dir, f"opt_{self.identifier}_{epoch:06d}.pt"),
                "wb",
            ) as f:
                torch.save(opt.state_dict(), f)

        dist.barrier()

    def udpate_ema(self):
        def update_ema(target_params, source_params, rate=0.99):
            """
            Update target parameters to be closer to those of source parameters using
            an exponential moving average.

            :param target_params: the target parameter sequence.
            :param source_params: the source parameter sequence.
            :param rate: the EMA rate (closer to 1 means slower).
            """
            for targ, src in zip(target_params, source_params):
                targ.detach().mul_(rate).add_(src, alpha=1 - rate)

        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model_params, rate=rate)

    def _master_params_to_state_dict(self, master_params):
        # if self.use_fp16:
        #     master_params = unflatten_master_params(
        #         self.model.parameters(), master_params
        #     )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        # if self.use_fp16:
        #     return make_master_params(params)
        # else:
        return params


    def _load_optimizer_state(self, opt):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt_{self.identifier}_{self.resume_epoch:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            self._log(f"{self.identifier} - loading vf optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            opt.load_state_dict(state_dict)
        return opt


def find_ema_checkpoint_vf(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_vf_predictor_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
        
def parse_resume_step_from_filename_vf(filename):
    """
    Parse filenames of the form path/to/vf_predictor_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("vf_predictor_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def find_ema_checkpoint_(main_checkpoint, step, rate, identifier):
    if main_checkpoint is None:
        return None
    filename = f"ema_{identifier}_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def parse_resume_step_from_filename_(filename, identifier='predictor'):
    """
    Parse filenames of the form path/to/vf_predictor_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split(f"{identifier}_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())

def classify_glaucoma(mds, num_class=2, th=-1):
    # glau = np.where(mds<-3.0, np.ones_like(mds), np.zeros_like(mds))
    if num_class==3:
        borderline = np.where((mds>=-3.0) & (mds<-1.0), np.ones_like(mds), np.zeros_like(mds))
        non_glau = np.where(mds>=-1.0, np.ones_like(mds), np.zeros_like(mds))
        y_pred = borderline + non_glau * 2
    elif num_class==2:
        y_pred = np.where(mds>=th, np.ones_like(mds), np.zeros_like(mds))
    return y_pred

def classify(prob):
    y = (prob>=0.5).astype(float)
    return y

def to_one_hot_vector(arr):
    arr = arr.astype(int)
    shape = (arr.shape[0], arr.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(arr.shape[0])
    one_hot[rows, arr] = 1
    return one_hot

def compute_weight(weight, step, rampup_step=4000):
    return weight if rampup_step == 0 or step > rampup_step else weight * step / rampup_step

#=====> lr schedulers
# https://github.com/godofpdog/ViT_PyTorch/blob/af086058764e55a48043db7f6f7c32b685db9427/vit_pytorch/solver.py
def get_scheduler(optimizer, args):
    if args.scheduler is not None:
        if args.scheduler == 'cosine':
            return CosineAnnealingLR(optimizer, args.t_max, args.eta_min)
        elif args.scheduler == 'step':
            return StepLR(optimizer, args.step_size, args.gamma)
        elif args.scheduler == 'exp':
            return ExponentialLR(optimizer, args.gamma)
        else:
            raise ValueError('Invalid scheduler.')
    else:
        return ConstantScheduler(optimizer)

# https://github.com/jeonsworld/ViT-pytorch/blob/main/utils/scheduler.py

class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)

class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
#<===== lr schedulers