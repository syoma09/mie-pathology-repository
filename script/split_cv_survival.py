#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd


def count_subject_image(df):
    """
    Count number of images corresponding to a subject
    :param df:
    :return:
    """

    subjects = [
        subject.split('-')[0]
        for subject in df.number
    ]

    counts = {
        subject: subjects.count(subject)
        for subject in subjects
    }

    return counts


def split_cv(counts, cv=4):
    counts = [
        (subject, count)
        for subject, count in counts.items()
    ]

    counts = sorted(counts, key=lambda x: x[1], reverse=True)
    print(counts)

    # # Reverse order of even chunks
    # def chunk_rev_range(i):
    #     if i // cv % 2 == 0:
    #         return range(i, i + cv)
    #     else:
    #         return range(i + cv - 1, i - 1, -1)
    # counts = [
    #     counts[ci]
    #     for i in range(0, len(counts), cv)
    #     for ci in chunk_rev_range(i)
    #     if ci < len(counts)
    # ]
    # print(counts)

    cv_dataset = [[[], 0] for _ in range(cv)]
    for data in counts:
        idx = np.argmin([n for [_, n] in cv_dataset])

        cv_dataset[idx][0].append(data)
        cv_dataset[idx][1] += data[1]

    for [d, _] in cv_dataset:
        print(d, sum(n for _, n in d))

    return [
        dataset for [dataset, _] in cv_dataset
    ]


def main():
    # Source annotation file
    src = Path(
        # "~/workspace/mie-pathology/_data/survival_cls2.csv"
        # "~/workspace/mie-pathology/_data/pick_up.csv"
        # "../_data/20220413.csv"
        # "../_data/20220413s.csv"
        "../_data/20220428_3os.csv"
    ).expanduser().absolute()
    # Destination root
    dst = src.parent / src.stem
    dst.mkdir(parents=True, exist_ok=True)

    # Load annotation
    df = pd.read_csv(src)

    # Retrieve non-survivers
    # df_neg = df[df.label == 0]
    df_neg = df[df.OS == 0]
    cv_neg = split_cv(count_subject_image(df_neg))

    # Retrieve survivers
    # df_pos = df[df.label == 1]
    df_pos = df[df.OS == 1]
    cv_pos = split_cv(count_subject_image(df_pos))

    # Sort
    def sort_key(dataset):
        return sum(
            n for _, n in dataset
        )
    cv_neg = sorted(cv_neg, key=sort_key)
    cv_pos = sorted(cv_pos, key=sort_key, reverse=True)

    for cv, (valid_neg, valid_pos) in enumerate(zip(cv_neg, cv_pos)):
        # Set all data as training (tvt==0)
        df['tvt'] = 0
        for subject, _ in valid_neg + valid_pos:
            df.loc[
                df['number'].str.startswith(subject), 'tvt'
            ] = 1

        print("##################################################")
        print(df)
        df.to_csv(
            dst / f"cv{cv}.csv", index=False
        )


if __name__ == '__main__':
    main()
