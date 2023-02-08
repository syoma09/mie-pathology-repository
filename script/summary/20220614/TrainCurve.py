#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Sans Serif"


def get_tb_value(log: str, tag: str) -> [float]:
    event = EventAccumulator(log)
    event.Reload()
    scalars = [
        v.value for v in event.Scalars(tag)
    ]

    return scalars


def save_plot(name: str):
    plt.savefig(f"{name}.jpg")
    plt.savefig(f"{name}.pdf")
    plt.close()


def main():
    logs = {
        "3OS": {
            "cv0": "./events.out.tfevents.1655271087.reaper.6757.0",        # 20220615_143124
            "cv1": "./events.out.tfevents.1655271079.antimon.89375.0",      # 20220615_143117
            "cv2": "./events.out.tfevents.1655273247.predator.3972988.0",   # 20220615_150709
            "cv3": "./events.out.tfevents.1655344963.predator.1954344.0"    # 20220616_110237
        }, "3MFS": {
            "cv0": "./events.out.tfevents.1655436269.reaper.925116.0",      # 20220617_122426
            "cv1": "./events.out.tfevents.1655372074.antimon.298037.0",     # 20220616_183431
            "cv2": "./events.out.tfevents.1655699316.antimon.502048.0",     # 20220620_132834
            "cv3": "./events.out.tfevents.1655699190.predator.3455101.0"    # 20220620_132625
        }
    }

    """3-OS: Training loss"""
    outcome, dataset, metric = "3OS", "train", "loss"
    plt.plot(get_tb_value(logs[outcome]["cv0"], f"{dataset}_{metric}"), label='CV1', linestyle='solid')
    plt.plot(get_tb_value(logs[outcome]["cv1"], f"{dataset}_{metric}"), label='CV2', linestyle='dashed')
    plt.plot(get_tb_value(logs[outcome]["cv2"], f"{dataset}_{metric}"), label='CV3', linestyle='dashdot')
    plt.plot(get_tb_value(logs[outcome]["cv3"], f"{dataset}_{metric}"), label='CV4', linestyle='dotted')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.xlim(-1, 301)
    plt.ylim(0, 0.05)

    plt.legend()
    plt.tight_layout()

    save_plot(f"{outcome}_{dataset}_{metric}")

    """: Training F-measure"""
    outcome, dataset, metric = "3OS", "train", "f1"
    plt.plot(get_tb_value(logs[outcome]["cv0"], f"{dataset}_{metric}"), label='CV1', linestyle='solid')
    plt.plot(get_tb_value(logs[outcome]["cv1"], f"{dataset}_{metric}"), label='CV2', linestyle='dashed')
    plt.plot(get_tb_value(logs[outcome]["cv2"], f"{dataset}_{metric}"), label='CV3', linestyle='dashdot')
    plt.plot(get_tb_value(logs[outcome]["cv3"], f"{dataset}_{metric}"), label='CV4', linestyle='dotted')

    plt.xlabel("Epoch")
    plt.ylabel("F-measure")

    plt.xlim(-1, 301)
    plt.ylim(0, 1.1)

    plt.legend()
    plt.tight_layout()

    save_plot(f"{outcome}_{dataset}_{metric}")

    """3OS: Validation loss"""
    outcome, dataset, metric = "3OS", "valid", "loss"
    plt.plot(get_tb_value(logs[outcome]["cv0"], f"{dataset}_{metric}"), label='CV1', linestyle='solid')
    plt.plot(get_tb_value(logs[outcome]["cv1"], f"{dataset}_{metric}"), label='CV2', linestyle='dashed')
    plt.plot(get_tb_value(logs[outcome]["cv2"], f"{dataset}_{metric}"), label='CV3', linestyle='dashdot')
    plt.plot(get_tb_value(logs[outcome]["cv3"], f"{dataset}_{metric}"), label='CV4', linestyle='dotted')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.xlim(-1, 301)
    plt.ylim(0, 10)

    plt.legend()
    plt.tight_layout()

    save_plot(f"{outcome}_{dataset}_{metric}")

    """3OS: Validation F-measure"""
    outcome, dataset, metric = "3OS", "valid", "f1"
    plt.plot(get_tb_value(logs[outcome]["cv0"], f"{dataset}_{metric}"), label='CV1', linestyle='solid')
    plt.plot(get_tb_value(logs[outcome]["cv1"], f"{dataset}_{metric}"), label='CV2', linestyle='dashed')
    plt.plot(get_tb_value(logs[outcome]["cv2"], f"{dataset}_{metric}"), label='CV3', linestyle='dashdot')
    plt.plot(get_tb_value(logs[outcome]["cv3"], f"{dataset}_{metric}"), label='CV4', linestyle='dotted')

    plt.xlabel("Epoch")
    plt.ylabel("F-measure")

    plt.xlim(-1, 301)
    plt.ylim(0, 1.1)

    plt.legend()
    plt.tight_layout()
    save_plot(f"{outcome}_{dataset}_{metric}")

    """3MFS: Training Loss"""
    outcome, dataset, metric = "3MFS", "train", "loss"
    plt.plot(get_tb_value(logs[outcome]["cv0"], f"{dataset}_{metric}"), label='CV1', linestyle='solid')
    plt.plot(get_tb_value(logs[outcome]["cv1"], f"{dataset}_{metric}"), label='CV2', linestyle='dashed')
    plt.plot(get_tb_value(logs[outcome]["cv2"], f"{dataset}_{metric}"), label='CV3', linestyle='dashdot')
    plt.plot(get_tb_value(logs[outcome]["cv3"], f"{dataset}_{metric}"), label='CV4', linestyle='dotted')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.xlim(-1, 301)
    plt.ylim(0, 0.1)

    plt.legend()
    plt.tight_layout()
    save_plot(f"{outcome}_{dataset}_{metric}")

    """3MFS: Training F-measure"""
    outcome, dataset, metric = "3MFS", "train", "f1"
    plt.plot(get_tb_value(logs[outcome]["cv0"], f"{dataset}_{metric}"), label='CV1', linestyle='solid')
    plt.plot(get_tb_value(logs[outcome]["cv1"], f"{dataset}_{metric}"), label='CV2', linestyle='dashed')
    plt.plot(get_tb_value(logs[outcome]["cv2"], f"{dataset}_{metric}"), label='CV3', linestyle='dashdot')
    plt.plot(get_tb_value(logs[outcome]["cv3"], f"{dataset}_{metric}"), label='CV4', linestyle='dotted')

    plt.xlabel("Epoch")
    plt.ylabel("F-measure")

    plt.xlim(-1, 301)
    plt.ylim(0, 1.1)

    plt.legend()
    plt.tight_layout()
    save_plot(f"{outcome}_{dataset}_{metric}")

    """3MFS: Valiation Loss"""
    outcome, dataset, metric = "3MFS", "valid", "loss"
    plt.plot(get_tb_value(logs[outcome]["cv0"], f"{dataset}_{metric}"), label='CV1', linestyle='solid')
    plt.plot(get_tb_value(logs[outcome]["cv1"], f"{dataset}_{metric}"), label='CV2', linestyle='dashed')
    plt.plot(get_tb_value(logs[outcome]["cv2"], f"{dataset}_{metric}"), label='CV3', linestyle='dashdot')
    plt.plot(get_tb_value(logs[outcome]["cv3"], f"{dataset}_{metric}"), label='CV4', linestyle='dotted')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.xlim(-1, 301)
    plt.ylim(0, 10)

    plt.legend()
    plt.tight_layout()
    save_plot(f"{outcome}_{dataset}_{metric}")

    """3MFS: Validation F-measure"""
    outcome, dataset, metric = "3MFS", "valid", "f1"
    plt.plot(get_tb_value(logs[outcome]["cv0"], f"{dataset}_{metric}"), label='CV1', linestyle='solid')
    plt.plot(get_tb_value(logs[outcome]["cv1"], f"{dataset}_{metric}"), label='CV2', linestyle='dashed')
    plt.plot(get_tb_value(logs[outcome]["cv2"], f"{dataset}_{metric}"), label='CV3', linestyle='dashdot')
    plt.plot(get_tb_value(logs[outcome]["cv3"], f"{dataset}_{metric}"), label='CV4', linestyle='dotted')

    plt.xlabel("Epoch")
    plt.ylabel("F-measure")

    plt.xlim(-1, 301)
    plt.ylim(0, 1.1)

    plt.legend()
    plt.tight_layout()
    save_plot(f"{outcome}_{dataset}_{metric}")

if __name__ == '__main__':
    main()
