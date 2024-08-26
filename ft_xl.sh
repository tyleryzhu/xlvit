PRETRAIN_CHKPT=runs_xl25/checkpoint-250.pth
JOB_DIR=/home/tyler/xl_mae/runs_ft_xl25
IMAGENET_DIR=/home/group/ilsvrc
python submitit_finetune.py \
    --job_dir ${JOB_DIR} \
    --nodes 1 \
    --ngpus 2 \
    --nodelist em4 \
    --qos low \
    --wandb ft_xl25 \
    --batch_size 324 \
    --model vit_tiny_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --global_pool \
    --epochs 100 \
    --blr 1e-3 --layer_decay 0.85 \
    --smoothing 0.0 --aa rand-m10-mstd0.5-inc1 \
    --weight_decay 0.05 --drop_path 0.0 --reprob 0.25 --mixup 0.2 --cutmix 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
