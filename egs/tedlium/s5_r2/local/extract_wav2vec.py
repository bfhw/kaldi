#!/usr/bin/env python3
# Extract wav2vec (2.0) features from wav files and segments
# Natalia Tomashenko

import torch
import fairseq
import argparse
import os
import numpy as np
from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Model
from torch import nn
import torch.nn.functional as F

from kaldiio import WriteHelper, ReadHelper
from tqdm import tqdm


class PretrainedWav2VecModel(nn.Module):
    def __init__(self, fname):
        super().__init__()

        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([fname])
        self.normalize = ("normalize" in cfg["task"]) and cfg["task"]["normalize"]
        model = model[0]
        model.eval()
        self.model = model

    def forward(self, x):
        with torch.no_grad():
            # self.model.eval()
            if self.normalize:
                x = F.layer_norm(x, x.shape)
            return self.model.extract_features(x, padding_mask=None, mask=False)


class Prediction:
    def __init__(self, fname, gpu=0):
        self.gpu = gpu
        self.model = PretrainedWav2VecModel(fname).cuda(gpu)

    def __call__(self, x):
        x = torch.from_numpy(x).float().cuda(self.gpu)
        with torch.no_grad():
            # f, padding = self.model(x.unsqueeze(0))
            r = self.model(x.unsqueeze(0))
            f = r["x"]

            def normalize_layer_result(layer_result):
                x, _, _ = layer_result
                x = x.transpose(0, 1)
                x = F.layer_norm(x, (1024,))

                return x.squeeze(0).cpu().numpy()

            def normalize_feature(x):
                x = F.layer_norm(x, (512,))
                return x.squeeze(0).cpu().numpy()

            # layer_result_24 = normalize_layer_result(r["layer_results"][23])
            # layer_result_19 = normalize_layer_result(r["layer_results"][18])
            layer_result_11 = normalize_layer_result(r["layer_results"][10])
            # layer_result_22 = normalize_layer_result(r["layer_results"][21])
            # features = normalize_feature(r["features"])
            layer_result_2 = normalize_layer_result(r["layer_results"][1])

        # return f.squeeze(0).cpu().numpy()
        # return layer_result_24, layer_result_19, layer_result_22, features
        return layer_result_11, layer_result_2
        


def write_features(model, input, output):
    os.makedirs(output, exist_ok=True)

    with open(f"{input}/segments") as f:
        n = sum(1 for _ in f)

    pbar = tqdm(total=n, desc="w2v feat extraction")

    with ReadHelper(f'ark:extract-segments scp:{input}/wav.scp {input}/segments ark:-|') as reader:
    # with ReadHelper(f'scp:{input}/wav.scp', segments=f'{input}/segments') as reader:
    # with ReadHelper(f"scp:{input}/wav.scp") as reader:
        with WriteHelper(f"ark,scp:{output}/feats.ark,{output}/feats.scp") as writer:
            for key, (sf, wav) in reader:
                wav = wav.astype(dtype=np.float32)
                feat = model(wav)
                feat = np.repeat(feat, 2, axis=0)
                writer(key, feat)

                pbar.update(1)

    pbar.close()


# def write_features_multi(model, input, output_24, output_19, output_22, output_feat):
def write_features_multi(model, input, output_11, output_2):
    # os.makedirs(output_24, exist_ok=True)
    # os.makedirs(output_19, exist_ok=True)
    # os.makedirs(output_22, exist_ok=True)
    # os.makedirs(output_feat, exist_ok=True)

    os.makedirs(output_11, exist_ok=True)
    os.makedirs(output_2, exist_ok=True)

    with open(f"{input}/segments") as f:
        n = sum(1 for _ in f)

    pbar = tqdm(total=n, desc="w2v feat extraction")

    with ReadHelper(f'ark:extract-segments scp:{input}/wav.scp {input}/segments ark:-|') as reader:
    # with ReadHelper(f"scp:{input}/wav.scp") as reader:
        # with WriteHelper(f"ark,scp:{output_24}/feats.ark,{output_24}/feats.scp") as writer_24,  WriteHelper(f"ark,scp:{output_19}/feats.ark,{output_19}/feats.scp") as writer_19,  WriteHelper(f"ark,scp:{output_22}/feats.ark,{output_22}/feats.scp") as writer_22,  WriteHelper(f"ark,scp:{output_feat}/feats.ark,{output_feat}/feats.scp") as writer_feat:
        with WriteHelper(f"ark,scp:{output_11}/feats.ark,{output_11}/feats.scp") as writer_11,  WriteHelper(f"ark,scp:{output_2}/feats.ark,{output_2}/feats.scp") as writer_2:
            for key, (sf, wav) in reader:
                wav = wav.astype(dtype=np.float32)
                # feat = model(wav)
                # feat = np.repeat(feat, 2, axis=0)
                # writer(key, feat)

                results = model(wav)
                # layer_result_24, layer_result_19, layer_result_22, features = (np.repeat(feat, 2, axis=0) for feat in results)
                layer_result_11, layer_result_2 = (np.repeat(feat, 2, axis=0) for feat in results)

                # writer_24(key, layer_result_24)
                # writer_19(key, layer_result_19)
                # writer_22(key, layer_result_22)
                # writer_feat(key, features)

                writer_11(key, layer_result_11)
                writer_2(key, layer_result_2)

                pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Usage: extract_wav2vec.py <model> <input> <output>")
    parser.add_argument("model", type=str)
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    print("Options:")
    print("  model: {}".format(args.model))
    print("  input: {}".format(args.input))
    print("  output: {}".format(args.output))

    print("  Loading model...")
    model = Prediction(args.model)

    # import soundfile as sf

    # wav_path = "/home/bhuang/tmp/sample_16k_1s.wav"
    # wavform, sr = sf.read(wav_path, dtype="float32")
    # print(wavform.shape)

    # feat = model(wavform)
    # print(feat)
    # print(feat.shape)
    # # ? why duplicate
    # feat = np.repeat(feat, 2, axis=0)
    # print(feat)
    # print(feat.shape)
    # quit()

    print("  Writing Features...")
    # write_features(model, args.input, args.output)
    # write_features_multi(model, args.input, f"{args.input}_w2v_layer24", f"{args.input}_w2v_layer19", f"{args.input}_w2v_layer22", f"{args.input}_w2v_features")
    write_features_multi(model, args.input, f"{args.input}_w2v_layer11", f"{args.input}_w2v_layer2")
    print("Done.")
