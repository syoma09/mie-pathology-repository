#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt


def get_tb_value(log: Path, tag: str) -> [float]:
    return [
        v.simple_value
        for e in tf.compat.v1.train.summary_iterator(str(log))
        for v in e.summary.value
        if v.tag == tag
    ]


def plot_valid_loss(log: Path):
    data = get_tb_value(log, 'valid_loss')
    plt.plot(
        list(range(len(data))), data
    )
    plt.xlabel("Epoch")
    plt.ylabel("BCE")
    plt.title("Validation Loss")

    plt.xlim(0, (len(data) + 50) // 50 * 50)
    plt.ylim(
        int(min(data) - 1),
        int(max(data) + 1)
    )

    plt.tick_params(direction='in')
    plt.grid()

    plt.tight_layout()
    plt.savefig("valid_loss.jpg")
    # plt.show()
    plt.close()


def plot_valid_f1(log: Path):
    tag = 'valid_f1'

    data = get_tb_value(log, tag)
    plt.plot(
        list(range(len(data))), data
    )
    plt.xlabel("Epoch")
    plt.ylabel("f-measure")
    plt.title("Validation f-measure")

    plt.xlim(0, (len(data) + 50) // 50 * 50)
    plt.ylim(0, 1)

    plt.tick_params(direction='in')
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{tag}.jpg")
    # plt.show()
    plt.close()


def plot_train_loss(log: Path):
    data = get_tb_value(log, 'train_loss')
    plt.plot(
        list(range(len(data))), data
    )
    plt.xlabel("Epoch")
    plt.ylabel("BCE")
    plt.title("Training Loss")

    plt.xlim(0, (len(data) + 50) // 50 * 50)
    plt.ylim(0, None)

    plt.tick_params(direction='in')
    plt.grid()

    plt.tight_layout()
    plt.savefig("train_loss.jpg")
    # plt.show()
    plt.close()


def plot_train_f1(log: Path):
    tag = "train_f1"
    data = get_tb_value(log, tag)
    plt.plot(
        list(range(len(data))), data
    )
    plt.xlabel("Epoch")
    plt.ylabel("f-measure")
    plt.title("Training f-measure")

    plt.xlim(0, (len(data) + 50) // 50 * 50)
    plt.ylim(0.8, 1)

    plt.tick_params(direction='in')
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{tag}.jpg")
    # plt.show()
    plt.close()


def main():
    tf_log = Path(
        "../logs/events.out.tfevents.1625215906.triton.706654.0"
    ).absolute()

    plot_valid_loss(tf_log)
    plot_valid_f1(tf_log)
    plot_train_loss(tf_log)
    plot_train_f1(tf_log)


if __name__ == '__main__':
    main()
