SKIP_FIRST=100
JOB_DIR=/home/tyler/xl_mae/runs_xl_skip${SKIP_FIRST}
mkdir -p $JOB_DIR
python submitit_pretrain.py \
    --accum_iter 4 \
    --job_dir ${JOB_DIR} \
    --nodes 1 \
    --ngpus 3 \
    --nodelist em3 \
    --qos medium \
    --batch_size 180 \
    --model xl_vit_tiny_patch16 \
    --skip_first $SKIP_FIRST \
    --norm_pix_loss \
    --num_workers 10 \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /home/group/ilsvrc