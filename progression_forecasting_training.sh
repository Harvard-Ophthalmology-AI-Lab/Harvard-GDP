#!/bin/bash
EXPR=progression_forecasting # set this for grouping curves on wandb.ai
MODEL_TYPE='efficientnet' # efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
LOSS_TYPE='bce' # mse | cos | kld | mae | gaussnll | bce 
LR=2e-5 
BATCH_SIZE=6
MODALITY=2 # 1: only RNFLT, 2: RNFLT + VF's tds
PROGRESSION_TYPE=progression_outcome_md_fast_no_p_cut # progression_outcome_td_pointwise_no_p_cut | progression_outcome_md_fast_no_p_cut
LR_POL=1e-4
TIMEWINDOW=50
DISCOUNT=0.9
# Only using labeled samples
python scripts/train_progression_pseudo_supervisor.py \
		--data_dir /path-to-progression-forecasting-dataset/ \
		--log_dir ./results/Longitudinal/${MODEL_TYPE}_lr${LR}_bz${BATCH_SIZE}_${MODALITY}modality_${PROGRESSION_TYPE}_labeled \
		--model ${MODEL_TYPE} \
		--image_size 224 \
		--loss_type ${LOSS_TYPE} \
		--lr_vf ${LR} --weight_decay_vf 0. \
		--lr_policy ${LR_POL} \
		--batch_size ${BATCH_SIZE} \
		--num_epochs 10 \
		--experiment_name ${EXPR} \
		--data_type label \
		--progression_outcome ${PROGRESSION_TYPE} \
		--semi_reinforce_beta ${TIMEWINDOW} \
        --semi_reinforce_gamma ${DISCOUNT} \
		--data_modality ${MODALITY} 
# Using labeled and unlabeled samples
python scripts/train_progression_pseudo_supervisor.py \
		--data_dir /path-to-progression-forecasting-dataset/ \
		--log_dir ./results/Longitudinal/${MODEL_TYPE}_lr${LR}_bz${BATCH_SIZE}_${MODALITY}modality_${PROGRESSION_TYPE}_labeled_and_unlabeled \
		--model ${MODEL_TYPE} \
		--image_size 224 \
		--loss_type ${LOSS_TYPE} \
		--lr_vf ${LR} --weight_decay_vf 0. \
		--lr_policy ${LR_POL} \
		--batch_size ${BATCH_SIZE} \
		--num_epochs 10 \
		--experiment_name ${EXPR} \
		--data_type label+unlabel \
		--progression_outcome ${PROGRESSION_TYPE} \
		--semi_reinforce_beta ${TIMEWINDOW} \
        --semi_reinforce_gamma ${DISCOUNT} \
		--data_modality ${MODALITY} 