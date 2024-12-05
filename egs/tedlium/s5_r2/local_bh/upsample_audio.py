#!/usr/bin/env python
# coding=utf-8
# Copyright 2021  Bofeng Huang

"""Upsampling with librosa"""

import json
import os
import sys
import subprocess
import fire

import librosa
import soundfile as sf


def load_upsample_write(in_wav_path, out_wav_path, in_sr, out_sr, offset=0, duration=None):
    # cmd = ["sox", ]
    # subprocess.run(cmd)

    y, _ = librosa.load(in_wav_path, sr=in_sr, offset=offset, duration=duration)
    y_upsampled = librosa.resample(y, orig_sr=in_sr, target_sr=out_sr)
    # obsolete
    # librosa.output.write_wav(out_wav_path, y_upsampled, out_sr)
    sf.write(out_wav_path, y_upsampled, out_sr, "PCM_16")


if __name__ == "__main__":
    fire.Fire(load_upsample_write)