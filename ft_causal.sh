PRETRAIN_CHKPT=ckpts/causal.pth
lr=$1
JOB="ft_causal_lr${lr}"
JOB_DIR=/home/tyler/xl_mae/runs_${JOB}
IMAGENET_DIR=/home/group/ilsvrc
python submitit_finetune.py \
    --accum_iter 2 \
    --job_dir ${JOB_DIR} \
    --wandb ${JOB} \
    --nodes 1 \
    --ngpus 4 \
    --qos low \
    --batch_size 168 \
    --model vit_tiny_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --global_pool \
    --epochs 100 \
    --blr $lr --layer_decay 0.85 \
    --smoothing 0.0 --aa rand-m10-mstd0.5-inc1 \
    --weight_decay 0.05 --drop_path 0.0 --reprob 0.25 --mixup 0.2 --cutmix 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}

