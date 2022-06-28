#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def get_tb_value(log: Path, tag: str) -> [float]:
    return [
        v.simple_value
        for e in tf.compat.v1.train.summary_iterator(str(log))
        for v in e.summary.value
        if v.tag == tag
    ][:300]
    # return [
    #     v.simple_value
    #     for e in tf.data.TFRecordDataset(log)
    #     for v in e.summary.value
    #     if v.tag == tag
    # ]


def plot_loss(log: Path, dataset):
    data = get_tb_value(log, f"{dataset}_loss")
    plt.plot(
        list(range(len(data))), data
    )
    plt.xlabel("Epoch")
    plt.ylabel("BCE")
    plt.title(f"{dataset} Loss")

    plt.xlim(0, (len(data) + 50) // 50 * 50)
    plt.ylim(
        int(min(data) - 1),
        int(max(data) + 1)
    )

    plt.tick_params(direction='in')
    plt.grid()

    plt.tight_layout()
    plt.savefig((log / "..").resolve() / f"{dataset}_loss.jpg")
    plt.savefig((log / "..").resolve() / f"{dataset}_loss.pdf")

    plt.close()


def plot_metric(path: Path, dataset, metric_type):
    tag = f"{dataset}_{metric_type}"
    data = get_tb_value(path, tag)
    print("  max:", max(data), "@", np.argmax(data))

    plt.plot(
        list(range(len(data))), data
    )
    plt.xlabel("Epoch")
    plt.ylabel(metric_type)
    plt.title(f"{dataset} {metric_type}")

    plt.xlim(0, (len(data) + 50) // 50 * 50)
    plt.ylim(0, 1)
    # plt.ylim(0, None)

    plt.tick_params(direction='in')
    plt.grid()

    plt.tight_layout()
    plt.savefig((path / "..").resolve() / f"{tag}.jpg")
    plt.savefig((path / "..").resolve() / f"{tag}.pdf")

    plt.close()


def main():
    tf_log = Path("~/data/_out/mie-pathology").expanduser()

    # # Find latest logging directory
    # tf_log = sorted(tf_log.iterdir())[-1]
    # Or manually select
    # tf_log /= "20220615_143124"     # 3OS-cv0
    # tf_log /= "20220615_143117"
    # tf_log /= "20220615_150709"
    # tf_log /= "20220616_110237"
    # tf_log /= "20220617_122426"   # 3MFS-cv0
    # tf_log /= "20220616_183431"
    # tf_log /= "20220620_132834"
    tf_log /= "20220620_132625"

    print("Process log in", tf_log)

    if tf_log.is_dir():
        tf_log = list(tf_log.glob("events.out.tfevents.*"))[0]

    # data = get_tb_value(tf_log, f"valid_f1")
    # print(data)
    # exit(0)

    for dataset in ['train', 'valid']:
        plot_loss(tf_log, dataset=dataset)

    metrics = {
        'train': ['f1inv', 'recall', 'precision'],
        'valid': ['f1', 'f1inv', 'recall', 'precision'],
    }
    for dataset, metric_types in metrics.items():
        for metric_type in metric_types:
            print(dataset, metric_type)
            plot_metric(tf_log, dataset, metric_type)


if __name__ == '__main__':
    main()
