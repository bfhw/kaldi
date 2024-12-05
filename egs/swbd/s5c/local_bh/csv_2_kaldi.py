#!/usr/bin/env python
# coding=utf-8


import logging
import os
import sys
import csv
from pathlib import Path
import numpy as np
import pandas as pd


def gen_kaldi_files(audio_path):
    return {
        "path": str(audio_path),
        "id": audio_path.stem,
        "spk": audio_path.stem,
    }


def main():
    csv_path = sys.argv[1]
    out_dir = sys.argv[2]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # df = pd.read_csv(csv_path)
    col_names = ["wav", "speaker", "id", "text", "start", "end"]
    df = pd.read_csv(csv_path, header=None, names=col_names)
    df["start"] = df["start"].astype(float)
    df.sort_values(["wav", "start"], inplace=True)
    # print(df.head(10))
    # quit()

    min_interval = 1
    data = []
    grouped = df.groupby("wav")
    for _, group in grouped:
        group.sort_values("start", inplace=True)

        group_new = group.to_dict('records')
        if not group_new:
            continue

        last = group_new[0]
        for i in range(1, len(group_new), 1):
            if group_new[i]["start"] - last["end"] < min_interval:
                last["text"] = last["text"] + " " + group_new[i]["text"]
                last["end"] =  group_new[i]["end"] 
            else:
                data.append(last)
                last = group_new[i]
        data.append(last)

    df = pd.DataFrame(data)
    # print(df.head(10))
    # quit()

    audio_dir = "/projects/bhuang/data/carglass_semi_annotated/wav"
    df["wav_path"] = df["wav"].map(lambda x: os.path.join(audio_dir, f"{x}.wav"))

    df["utt_id"] = df[["speaker", "id"]].apply(lambda x: "-".join(x), axis=1)

    df.sort_values("utt_id", inplace=True)


    df["start"] = df["start"].map(lambda x: max(0, x-0.5))
    # todo : total dur
    df["end"] = df["end"].map(lambda x: min(np.inf, x+0.5))

    # print(df.head())
    # quit()

    # sort
    # df.sort_values("id", inplace=True)
    # print(df.head())

    df[["utt_id", "speaker"]].to_csv(
        f"{out_dir}/utt2spk", sep=" ", quoting=csv.QUOTE_NONE, index=False, header=False
    )
    # df[["utt_id", "text"]].to_csv(
    #     f"{out_dir}/text", sep="\t", quoting=csv.QUOTE_NONE, index=False, header=False
    # )
    df[["wav", "wav_path"]].sort_values("wav").drop_duplicates().to_csv(
        f"{out_dir}/wav.scp", sep=" ", quoting=csv.QUOTE_NONE, index=False, header=False
    )
    df[["utt_id", "wav", "start", "end"]].to_csv(
        f"{out_dir}/segments", sep=" ", quoting=csv.QUOTE_NONE, index=False, header=False
    )

    with open(f"{out_dir}/text", "w") as f:
        s_ = df[["utt_id", "text"]].apply(lambda x: " ".join(x) + "\n", axis=1)
        f.writelines(s_.tolist())


if __name__ == "__main__":
    main()
