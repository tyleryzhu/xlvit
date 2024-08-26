JOB_DIR=/home/tyler/xl_mae/runs_xl/
CHKPT=runs_xl/checkpoint-80.pth
python submitit_pretrain.py \
    --accum_iter 4 \
    --job_dir ${JOB_DIR} \
    --resume ${CHKPT} \
    --nodes 1 \
    --ngpus 6 \
    --nodelist em3 \
    --qos medium \
    --batch_size 180 \
    --model xl_vit_tiny_patch16 \
    --norm_pix_loss \
    --epochs 300 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /home/group/ilsvrc
    # --use_volta32 i\