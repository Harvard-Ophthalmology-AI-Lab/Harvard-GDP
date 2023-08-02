import argparse
import json
import numpy as np

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torchvision.models import *
from torch.optim import *
import torch.nn.functional as F

from sklearn.metrics import *
import wandb

from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import *
from torch.distributions import Beta, Bernoulli

from utils.modules import *
from utils import logger
from utils.image_datasets import *


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir, log_suffix=args.data_subset)

    if args.random_seed < 0:
        args.random_seed = int(np.random.randint(10000, size=1)[0])
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    logger.log(f'===> random seed: {args.random_seed}')

    with open(os.path.join(args.log_dir, f'args_{args.data_subset}.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    wandb.init(
      project=args.experiment_name,
      config=vars(args),
    )
    wandb.run.name = f"{args.model}_{args.loss_type}_semimode{args.data_type}"
    wandb.run.save()

    # Initialize dataloaders for training, validation, and testing
    trn_dataset = CrossSectional_Dataset(args.data_dir, subset=args.data_subset, resolution=args.image_size)
    val_dataset = CrossSectional_Dataset(args.data_dir, subset=args.data_val_subset, resolution=args.image_size)
    tst_dataset = CrossSectional_Dataset(args.data_dir, subset=args.data_tst_subset, resolution=args.image_size)
    trn_dataloader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True, num_workers=2, drop_last=False)
    tst_dataloader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    val_dataiter = iter(cycle(val_dataloader))

    logger.log("creating VF predictor ...")

    predictor_head = nn.Sigmoid()

    vf_predictor = create_model(model_type=args.model, in_dim=1, out_dim=1)
    vf_predictor = nn.Sequential(vf_predictor, predictor_head)
    vf_predictor.to(dist_util.dev())
    vf_predictor = Model_Wrapper(vf_predictor, result_dir=args.log_dir, ema_rate=args.ema_rate_vf, 
                                    resume_checkpoint=args.resume_checkpoint_vf, logger=logger)

    pseudo_supervisor = create_model(model_type=args.model, in_dim=1, out_dim=1)
    pseudo_supervisor = nn.Sequential(pseudo_supervisor, nn.Sigmoid())
    pseudo_supervisor.to(dist_util.dev())
    pseudo_supervisor = Model_Wrapper(pseudo_supervisor, result_dir=args.log_dir, ema_rate=args.ema_rate_vf, 
                                    resume_checkpoint=args.resume_checkpoint_vf, logger=logger, identifier='policy')

    if args.loss_type == 'mse':
        loss_func = nn.MSELoss()
    elif args.loss_type == 'mae':
        loss_func = nn.L1Loss()
    elif args.loss_type == 'gaussnll':
        loss_func = nn.GaussianNLLLoss()
    elif args.loss_type == 'bce':
        loss_func = nn.BCELoss()

    optimizer = AdamW(vf_predictor.model_params, lr=args.lr_vf, betas=(0.0, 0.1), weight_decay=args.weight_decay_vf)
    optimizer = vf_predictor._load_checkpoint(optimizer)
    optimizer_policy = AdamW(pseudo_supervisor.model_params, lr=args.lr_policy, betas=(0.0, 0.1), weight_decay=args.weight_decay_vf)
    optimizer_policy = pseudo_supervisor._load_checkpoint(optimizer_policy)

    logger.log(f'training set includes {trn_dataset.__len__()} samples, validation set includes {val_dataset.__len__()} samples, test set includes {tst_dataset.__len__()} samples')

    num_steps = int(trn_dataset.__len__()/args.batch_size) * args.num_epochs
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=num_steps)
    
    trn_loss = AverageMeter()
    trn_mae = AverageMeter()
    val_loss = AverageMeter()
    val_mae = AverageMeter()
    
    state_pool = []
    action_pool = []
    reward_pool = []
    step_count = 0

    myepsilon = 1e-5
    best_perf = [sys.float_info.min]*4
    start_epoch = 0
    for epoch in range(start_epoch, args.num_epochs):
        # training
        trn_md_pred, trn_md_gt = np.empty(0), np.empty(0)

        vf_predictor.ddp_model.train()
        pseudo_supervisor.ddp_model.train()
        for step, (batch, batch_vf, _) in enumerate(trn_dataloader):
            # Fetch a mini-batch, which could include labeled and/or unlabeled samples
            val_batch, val_batch_gt, _ = next(val_dataiter)
            val_batch = val_batch.to(dist_util.dev())
            val_batch_gt = val_batch_gt.to(dist_util.dev())

            # Separate labeled and unlabeled samples
            batch = batch.to(dist_util.dev())
            mask = batch_vf<=-100
            labeled_mask = batch_vf >= 0
            unlabel_batch = None
            label_batch = None
            label_batch_vf = None
            if mask is not None and torch.any(mask):
                unlabel_batch = batch[mask,:,:,:]
            if labeled_mask is not None and torch.any(labeled_mask):
                label_batch = batch[labeled_mask,:,:,:]
                label_batch_vf = batch_vf[labeled_mask].to(dist_util.dev())
                if len(label_batch_vf.shape) == 0:
                    label_batch_vf = label_batch_vf.view(1)
            semi_loss = None

            pseudo_label = None
            with torch.no_grad():
                if unlabel_batch is not None:
                    smp_probs = pseudo_supervisor.ddp_model(unlabel_batch)
                    m = Bernoulli(smp_probs)
                    pseudo_label = m.sample().squeeze(1)
                    if len(pseudo_label.shape) == 0:
                        pseudo_label = pseudo_label.view(1)


            if label_batch is not None:
                tmp_batch = label_batch
                tmp_label = label_batch_vf
                labeled_ind = -1

                output = vf_predictor.ddp_model(tmp_batch).squeeze(1)
                md_pred = output
                md_gt = tmp_label

                md_pred_np = md_pred.detach().cpu().numpy()
                md_gt_np = md_gt.detach().cpu().numpy()

                optimizer.zero_grad()

                loss = loss_func(output, tmp_label)
                mae = np.abs(md_pred_np - md_gt_np).mean()

                loss.backward()
                optimizer.step()

                trn_loss.update(loss.item())
                trn_mae.update(mae.item())

                if not np.isnan(md_pred_np).any() and not np.isnan(md_gt_np).any():
                    trn_md_pred = np.concatenate((trn_md_pred, md_pred_np), axis=0)
                    trn_md_gt = np.concatenate((trn_md_gt, md_gt_np), axis=0)

            if unlabel_batch is not None:
                err_before = 0
                with torch.no_grad():
                    val_output = vf_predictor.ddp_model(val_batch).squeeze(1)
                    err_before = F.binary_cross_entropy(val_output, val_batch_gt).item()

                optimizer.zero_grad()
                output_ = vf_predictor.ddp_model(unlabel_batch).squeeze(1)
                loss_ = loss_func(output_, pseudo_label)

                loss_.backward()
                optimizer.step()
                scheduler.step()

                err_after = 0
                with torch.no_grad():
                    val_output = vf_predictor.ddp_model(val_batch).squeeze(1)
                    err_after = F.binary_cross_entropy(val_output, val_batch_gt).item()

                # Quantify empirical generalizability
                cur_reward = 0
                cur_reward += reward_function(err_before - err_after, args.semi_reinforce_reward_alpha)
                action_pool.append(pseudo_label)
                state_pool.append(unlabel_batch)
                reward_pool.append(cur_reward)

            if len(reward_pool)>1 and step % args.semi_reinforce_beta == 0:
                # Discount reward
                running_add = 0
                for i in range(len(reward_pool)-1, -1, -1):
                    if reward_pool[i] == 0:
                        running_add = 0
                    else:
                        running_add = running_add * args.semi_reinforce_gamma + reward_pool[i]
                        reward_pool[i] = running_add

                # Normalize reward
                reward_mean = np.mean(reward_pool)
                reward_std = np.std(reward_pool)
                for i in range(len(reward_pool)):
                    reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

                optimizer_policy.zero_grad()

                for i in range(len(reward_pool)):
                    state = state_pool[i]
                    action = action_pool[i] 
                    reward = reward_pool[i]

                    probs = pseudo_supervisor.ddp_model(state)
                    m = Bernoulli(probs)
                    rl_loss = (-m.log_prob(action) * reward).mean()
                    rl_loss.backward()

                optimizer_policy.step()
                state_pool = []
                action_pool = []
                reward_pool = []


        # Evaluate the performance
        pred_glau = classify(trn_md_pred)
        gt_glau = classify(trn_md_gt)

        trn_glau_acc = accuracy_score(gt_glau, pred_glau)
        trn_glau_macro_F1 = f1_score(gt_glau, pred_glau, average='macro')
        fpr, tpr, thresholds = roc_curve(trn_md_gt, trn_md_pred)
        trn_micro_AUC = auc(fpr, tpr)

        vf_predictor.save(epoch, optimizer)
        pseudo_supervisor.save(epoch, optimizer_policy)

        # validation
        val_md_pred, val_md_gt = np.empty(0), np.empty(0)
        vf_predictor.ddp_model.eval()
        pseudo_supervisor.ddp_model.eval()
        for step, (batch, batch_vf, _) in enumerate(tst_dataloader):
            batch = batch.to(dist_util.dev())
            batch_vf = batch_vf.to(dist_util.dev())
            output = vf_predictor.ddp_model(batch).squeeze(1)

            md_pred = output
            md_gt = batch_vf
            md_pred_np = md_pred.detach().cpu().numpy()
            md_gt_np = md_gt.detach().cpu().numpy()

            loss = loss_func(output, batch_vf)
            mae = np.abs(md_pred_np - md_gt_np).mean()

            val_loss.update(loss.item())
            val_mae.update(mae.item())

            if not np.isnan(md_pred_np).any() and not np.isnan(md_gt_np).any():
                val_md_pred = np.concatenate((val_md_pred, md_pred_np), axis=0)
                val_md_gt = np.concatenate((val_md_gt, md_gt_np), axis=0)


        # Evaluate the performance
        pred_glau = classify(val_md_pred)
        gt_glau = classify(val_md_gt)
        val_glau_acc = accuracy_score(gt_glau, pred_glau)
        val_glau_macro_F1 = f1_score(gt_glau, pred_glau, average='macro')
        fpr, tpr, thresholds = roc_curve(val_md_gt, val_md_pred)
        val_micro_AUC = auc(fpr, tpr)

        logger.logkv('epoch', epoch)
        logger.logkv('trn_loss', round(trn_loss.avg,4))
        logger.logkv('trn_glau_acc', round(trn_glau_acc,4))
        logger.logkv('trn_glau_macro_F1', round(trn_glau_macro_F1,4))
        logger.logkv('trn_micro_AUC', round(trn_micro_AUC,4))

        logger.logkv('val_loss', round(val_loss.avg,4))
        logger.logkv('val_glau_acc', round(val_glau_acc,4))
        logger.logkv('val_glau_macro_F1', round(val_glau_macro_F1,4))
        logger.logkv('val_micro_AUC', round(val_micro_AUC,4))

        logger.dumpkvs()

        if val_micro_AUC >= best_perf[-1]:
            best_perf[0] = epoch
            best_perf[1] = val_glau_acc
            best_perf[2] = val_glau_macro_F1
            best_perf[3] = val_micro_AUC
        
        wandb.log({"epoch": epoch, 
                        'trn_loss': trn_loss.avg,
                        'trn_acc': trn_glau_acc,
                        'trn_macro_F1': trn_glau_macro_F1,
                        'trn_micro_AUC': trn_micro_AUC,
                        'val_loss': val_loss.avg,
                        'val_acc': val_glau_acc,
                        'val_macro_F1': val_glau_macro_F1,
                        'val_micro_AUC': val_micro_AUC})

        str_output = f'epoch & Acc & macro-F1 & AUC \n'
        str_output += f'{best_perf[0]} & {best_perf[1]:.4f} & {best_perf[2]:.4f} & {best_perf[3]:.4f} \n'
        logger.log(str_output)

        trn_loss.reset()
        trn_mae.reset()
        val_loss.reset()
        val_mae.reset()

    os.rename(args.log_dir, f'{args.log_dir}_auc{best_perf[-1]:.4f}')


def create_argparser():
    defaults = dict(
        data_dir="",
        data_subset='train',
        data_val_subset='val',
        data_tst_subset='test',
        model='vit',
        schedule_sampler="uniform",
        random_seed=-1,
        image_size=224,
        lr_vf=1e-4,
        lr_policy=2e-5,
        weight_decay_vf=1e-5,
        ema_rate_vf="0.9999",
        resume_checkpoint_vf="",
        loss_type='mse',
        task='md',
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        val_batch_size=18,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        num_epochs=60,
        warmup_steps=0,
        data_type='label+unlabel',
        semi_reinforce_beta=50,
        semi_reinforce_gamma=0.9,
        semi_reinforce_reward_alpha=1.0,
        semi_reinforce_temperature=1.0,
        experiment_name="train_predictor_crosssectional_semi",
        glau_num_class=2,
        glau_threshold=-1,
        log_dir=''
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
