#/usr/bin/env python3
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
    ]   # [:100]
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
    plt.savefig(f"{dataset}_loss.jpg")

    plt.close()


def plot_metric(path: Path, dataset, metric_type):
    tag = f"{dataset}_{metric_type}"
    data = get_tb_value(path, tag)
    print(tag)
    print("  max:", max(data), "@", np.argmax(data))
    print("  min:", min(data), "@", np.argmin(data))
    plt.plot(
        list(range(len(data))), data
    )
    plt.xlabel("Epoch")
    plt.ylabel(metric_type)
    #plt.title(f"{dataset} {metric_type}")

    plt.xlim(0, (len(data) + 50) // 50 * 50)
    #(len(data) + 50) // 50 * 50
    #plt.ylim(0, 1)
    #plt.ylim(0, (max(data) + 50) // 50 * 50)
    plt.ylim(None, max(data))
    plt.tick_params(direction='in')
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{tag}.jpg")

    plt.close()


def main():
    tf_log = Path("~/root/workspace/mie-pathology/survival_time/logs").expanduser()

    # # Find latest logging directory
    # tf_log = sorted(tf_log.iterdir())[-1]
    # Or manually select
    #tf_log /= "20211228_w2014s512cv3_adam"

    print("Process log in", tf_log)

    if tf_log.is_dir():
        #tf_log = list(tf_log.glob("events.out.tfevents.17065254*"))[0]
        #tf_log = list(tf_log.glob("events.out.tfevents.17065255*"))[0]
        #tf_log = list(tf_log.glob("events.out.tfevents.1706868*"))[0]
        #tf_log = list(tf_log.glob("events.out.tfevents.170730565*"))[0]
        #tf_log = list(tf_log.glob("events.out.tfevents.1709711*"))[0] #AE tight
        #tf_log = list(tf_log.glob("events.out.tfevents.1709710*"))[0] #AE wise
        #tf_log = list(tf_log.glob("events.out.tfevents.1709709544*"))[0] #Trans tight
        #tf_log = list(tf_log.glob("events.out.tfevents.1709709599*"))[0] #Trans survival wise
        #tf_log = list(tf_log.glob("events.out.tfevents.1709709390*"))[0] #CL wise 
        tf_log = list(tf_log.glob("events.out.tfevents.17097086*"))[0] #CL tight


    # data = get_tb_value(tf_log, f"valid_f1")
    # print(data)
    # exit(0)

    #for dataset in ['train', 'valid']:
    #    plot_loss(tf_log, dataset=dataset)

    metrics = {
        #'train': ['loss','MV','MAE'],
        #'valid': ['loss','MV','MAE'],
        'train': ['MV','MAE','Index'],
        'valid':['MV','MAE','Index'],
    }
    for dataset, metric_types in metrics.items():
        for metric_type in metric_types:
            #print(dataset, metric_type)
            plot_metric(tf_log, dataset, metric_type)


if __name__ == '__main__':
    main()
