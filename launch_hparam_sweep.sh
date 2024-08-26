# bash script.sh $value $gpus $ckpt $wandb_id
# bash ft_xl_lr.sh 5e-4 em2 runs_ft_xl_lr5e-4/checkpoint-40.pth jyugm9by
# bash ft_xl_wd.sh 1e-2 em4 runs_ft_xl_wd1e-2/checkpoint-35.pth roqxfda5
# bash ft_xl_wd.sh 1e-3 em4 runs_ft_xl_wd1e-3/checkpoint-10.pth 24kevqfc
# bash ft_xl_wd.sh 0 em4 runs_ft_xl_wd0/checkpoint-35.pth 1uq3aw4n
# bash ft_xl_ld.sh 0.75 em4 runs_ft_xl_ld0.75/checkpoint-25.pth 3rs987pg
# bash ft_xl_ld.sh 0.65 em4 runs_ft_xl_ld0.65/checkpoint-75.pth 2zyxxlmp
# bash ft_xl_mixup.sh 0.1 em1 runs_ft_xl_mix0.1/checkpoint-25.pth 1hjqtj60
# bash ft_xl_smooth.sh 0.1 em1 runs_ft_xl_smooth0.1/checkpoint-35.pth 1utohllm
# bash ft_xl_good.sh runs_ft_xl_good/checkpoint-10.pth
# bash ft_xl_good2.sh runs_ft_xl_good2/checkpoint-4.pth

bash ft_causal_lr.sh 5e-4 em8 runs_ft_causal_lr5e-4/checkpoint-57.pth 20x0wkge
bash ft_causal_wd.sh 1e-3 em8 runs_ft_causal_wd1e-3/checkpoint-86.pth 11rmcst5
# bash ft_causal_wd.sh 0.0 em2 runs_ft_causal_wd0.0/checkpoint-59.pth lefie7sl
