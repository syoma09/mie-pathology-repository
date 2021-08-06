#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

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

    # Reverse order of even chunks
    def chunk_rev_range(i):
        if i // cv % 2 == 0:
            return range(i, i + cv)
        else:
            return range(i + cv - 1, i - 1, -1)
    counts = [
        counts[ci]
        for i in range(0, len(counts), cv)
        for ci in chunk_rev_range(i)
        if ci < len(counts)
    ]
    print(counts)

    cv_dataset = [
        counts[i::cv]
        for i in range(cv)
    ]
    for d in cv_dataset:
        print(d, sum(n for _, n in d))
    print(cv_dataset)


def main():
    df = pd.read_csv(Path(
        "~/workspace/mie-pathology/_data/survival_cls.csv"
    ).expanduser())

    # print(df)

    # Retrieve non-survivers
    df_neg = df[df.label == 0]
    # print(df_neg)
    split_cv(count_subject_image(df_neg))


if __name__ == '__main__':
    main()
