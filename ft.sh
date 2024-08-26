PRETRAIN_CHKPT=runs/checkpoint-240.pth
RESUME_CHKPT=runs_ft_tiny/checkpoint-19.pth
JOB_DIR=/home/tyler/xl_mae/runs_ft/
IMAGENET_DIR=/home/group/ilsvrc
python submitit_finetune.py \
    --accum_iter 4 \
    --job_dir ${JOB_DIR} \
    --nodes 1 \
    --ngpus 4 \
    --qos medium \
    --resume ${RESUME_CHKPT} \
    --batch_size 324 \
    --model vit_tiny_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 1e-3 --layer_decay 0.85 \
    --smoothing 0.0 --aa rand-m10-mstd0.5-inc1 \
    --weight_decay 0.05 --drop_path 0.0 --reprob 0.25 --mixup 0.2 --cutmix 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}