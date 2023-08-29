import argparse
import os

import optuna

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, default="0")
    parser.add_argument('--node', type=str, default="0015")
    opt = parser.parse_args()

    return opt

args = parse_args()

def evaluate_model(trial):
    # Clear the output file
    open("output.txt", "w").close()

    command = f"CUDA_VISIBLE_DEVICES=0,1,2,3 python /data/TTS_GAN/train_GAN.py \
        -gen_bs 16 \
        -dis_bs 16 \
        --dist-url 'tcp://localhost:4321' \
        --dist-backend 'nccl' \
        --world-size 1 \
        --rank {args.rank} \
        --dataset UniMiB \
        --bottom_width 8 \
        --max_iter 5000 \
        --img_size 32 \
        --gen_model my_gen \
        --dis_model my_dis \
        --df_dim 384 \
        --d_heads 4 \
        --d_depth 3 \
        --g_depth 5,4,2 \
        --dropout 0 \
        --latent_dim {trial.suggest_int('latent_dim', 100, 500, step=50)} \
        --gf_dim 1024 \
        --num_workers 1 \
        --g_lr 0.0001 \
        --d_lr 0.0003 \
        --optimizer adam \
        --loss lsgan \
        --wd 1e-3 \
        --beta1 0.9 \
        --beta2 0.999 \
        --phi 1 \
        --batch_size {trial.suggest_categorical('batch_size', [4, 8, 16])}\
        --num_eval_imgs 50000 \
        --init_type xavier_uniform \
        --n_critic 1 \
        --val_freq 20 \
        --print_freq 50 \
        --grow_steps 0 0 \
        --fade_in 0 \
        --patch_size {trial.suggest_categorical('patch_size', [10, 50, 100])}\
        --ema_kimg 500 \
        --ema_warmup 0.1 \
        --ema 0.9999 \
        --diff_aug translation,cutout,color \
        --class_name Running \
        --exp_name hp_tune \
        --embed_dim {trial.suggest_int('embed_dim', 10, 100, step=20)}"
    os.system(command)
    with open("output.txt", "r") as f:
        lines = f.readlines()
    l = lines[0].strip().split(' ')
    dtw = float(l[0])
    mse = float(l[1])
    cos_similarity = float(l[2])
    return mse

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(evaluate_model, n_trials=100)
    best_config = study.best_params
    with open("/data/TTS_GAN/optuna/best_config.txt", "w") as f:
        f.write(str(best_config))
    print(f"Best configuration found: {best_config}")
# ray.shutdown()
