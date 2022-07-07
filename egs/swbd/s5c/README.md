# SWBD-Based LVCSR

## Usage

*gmm*
```bash
# run bootstrapped gmm-hmm training
./run_gmm.sh
```

*nnet*
```bash
# set GPUs to exclusive mode to run kaldi nnet training
sudo nvidia-smi -i 2,3,4,5 -c 3

# sp
# result trained on hmhm 190h, tested on hmhm 10h, w/ LM lv hmhm merged
# %WER 11.17 [ 15187 / 135976, 1886 ins, 5783 del, 7518 sub ] (got some deaccented text in reference, real WER should be a bit lower)
CUDA_VISIBLE_DEVICES=2,3,4,5 ./run_tdnn_aug_7r.sh --speed_perturb true --multi_condition false

# sp + multi condition
# result trained on hmhm 190h, tested on hmhm 10h, w/ LM lv hmhm merged
# %WER 10.17 [ 13829 / 135976, 1861 ins, 5541 del, 6427 sub ]
CUDA_VISIBLE_DEVICES=2,3,4,5 ./run_tdnn_aug_7r.sh --speed_perturb true --multi_condition true
```
