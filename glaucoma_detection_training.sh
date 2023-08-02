#!/bin/bash
MODEL_TYPE='efficientnet' # efficientnet | vit | resnet | swin | vgg | resnext | wideresnet | efficientnetv1 | convnext
LOSS_TYPE='bce' # mse | cos | kld | mae | gaussnll | bce
LR=4e-5
EXPR=glaucoma_detection  # set this for grouping curves on wandb.ai
LR_POL=1e-4
BATCH_SIZE=12
TIMEWINDOW=50
DISCOUNT=0.9
# Only using labeled samples
python scripts/train_glaucoma_pseudo_supervisor.py \
		--data_dir /path-to-glaucoma-detection-dataset/ \
		--log_dir ./results/CrossSectional/${MODEL_TYPE}_${LOSS_TYPE}_lr${LR}_bz${BATCH_SIZE}_labeled \
		--model ${MODEL_TYPE} \
		--image_size 224 \
		--loss_type ${LOSS_TYPE} \
		--lr_vf ${LR} --weight_decay_vf 0.0 \
		--lr_policy ${LR_POL} \
		--batch_size ${BATCH_SIZE} \
		--num_epochs 10 \
		--experiment_name ${EXPR} \
		--data_type label \
		--semi_reinforce_beta ${TIMEWINDOW} \
		--semi_reinforce_gamma ${DISCOUNT}
# Using labeled and unlabeled samples
python scripts/train_glaucoma_pseudo_supervisor.py \
		--data_dir /path-to-glaucoma-detection-dataset/ \
		--log_dir ./results/CrossSectional/${MODEL_TYPE}_${LOSS_TYPE}_lr${LR}_bz${BATCH_SIZE}_labeled_and_unlabeled \
		--model ${MODEL_TYPE} \
		--image_size 224 \
		--loss_type ${LOSS_TYPE} \
		--lr_vf ${LR} --weight_decay_vf 0.0 \
		--lr_policy ${LR_POL} \
		--batch_size ${BATCH_SIZE} \
		--num_epochs 10 \
		--experiment_name ${EXPR} \
		--data_type label+unlabel \
		--semi_reinforce_beta ${TIMEWINDOW} \
		--semi_reinforce_gamma ${DISCOUNT}