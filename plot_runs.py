import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt

from tzutils.plot import quick_plot


def load_logs(file, tag):
    data = []
    with open(file, "r") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame.from_records(data)
    df["tag"] = tag
    return df


if __name__ == "__main__":
    file_mae = "runs/log.txt"
    file_mae_old = "runs/log_old.txt"
    file_causal = "runs_causal/log.txt"
    file_perm = "runs_perm/log.txt"
    file_xl = "runs_xl/old_log.txt"
    file_xl_s25 = "runs_xl_skip25/log.txt"
    file_xl_s100 = "runs_xl_skip100/log.txt"
    file_xl_s190 = "runs_xl_skip190/log.txt"

    file_ft_mae = "runs_ft/log.txt"
    file_ft_mae_sd = "runs_ft/log_em12.txt"
    file_ft_causal = "runs_ft_causal/log.txt"
    file_ft_xl = "runs_ft_xl/log.txt"
    df = pd.concat(
        [
            load_logs(file_mae_old, "loss_mae"),
            load_logs(file_mae, "loss_mae_small_decoder"),
            load_logs(file_causal, "loss_causal"),
            # load_logs(file_perm, "loss_perm"),
            load_logs(file_xl, "loss_xl_old"),
            load_logs(file_xl_s25, "loss_xl_skip25"),
            # load_logs(file_xl_s100, "loss_xl_skip100"),
            # load_logs(file_xl_s190, "loss_xl_skip190"),
        ]
    )

    # Pre-training loss
    plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    quick_plot(
        ax,
        df,
        "epoch",
        "train_loss",
        "tag",
        "ViT-Tiny on MAE Task",
        "Epoch",
        "Loss",
        smooth=False,
        smooth_weight=0.7,
    )
    plt.savefig(f"loss.png", bbox_inches="tight")

    df_xl = pd.concat(
        [
            load_logs(file_xl_s25, "loss_xl_skip25"),
            load_logs(file_xl_s100, "loss_xl_skip100"),
            load_logs(file_xl_s190, "loss_xl_skip190"),
        ]
    )
    plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    quick_plot(
        ax,
        df_xl,
        "epoch",
        "train_loss",
        "tag",
        "ViT-Tiny w/ XL PT Task, different skip amts",
        "Epoch",
        "Loss",
        smooth=False,
        smooth_weight=0.7,
    )
    plt.savefig(f"loss_xl.png", bbox_inches="tight")

    df_acc = pd.concat(
        [
            load_logs(file_ft_mae, "acc_mae"),
            load_logs(file_ft_mae_sd, "acc_mae_small_decoder"),
            load_logs(file_ft_causal, "acc_causal"),
            load_logs(file_ft_xl, "acc_xl"),
        ]
    )
    plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    quick_plot(
        ax,
        df_acc,
        "epoch",
        "test_acc1",
        "tag",
        "ViT-Tiny Imagenet Accuracy, Top-1",
        "Epoch",
        "Top-1 Test Accuracy",
        smooth=False,
        smooth_weight=0.7,
    )
    plt.savefig(f"acc_in1.png", bbox_inches="tight")
