JOB_DIR=/home/tyler/xl_mae/runs/
python submitit_pretrain.py \
    --job_dir ${JOB_DIR} \
    --nodes 1 \
    --ngpus 4 \
    --batch_size 120 \
    --model mae_vit_tiny_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /home/group/ilsvrc
    # --use_volta32 \