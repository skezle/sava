# Data subset selection by comparing to the validation set cia Optimal Tranport

This repository is the official implementation of the ICML 2024 submission "SAVA: Scalable Learning-Agnostic Data Valuation".  Do not distribute.

We propose SAVA: scalable model-agnostic algorithm for data valuation on labeled datasets by performing optimal transport hierarchically at both batch and data point levels.

This code base is forked from the [LAVA](https://github.com/ruoxi-jia-group/LAVA) we are immensely grateful to authors for open sourcing their code.



# Clothing1M

The dataset can be obtained by e-mailing the authors to obtain access.

## SAVA

## EL2N

## Supervised Prototypes

1. We need to generate the data's prototypes and score each data point by assigning it to it's custer center and calculating a cosine distance. We optimized the intra-cluster MSE to get the best number of clusters which was 10k.

```py
seed=4
python supervised_prototypes.py --cuda_num=3 --tag=test_lr005_${seed} --lr=0.05 \
    --seed=${seed} --n_cluster=10_000
```

1. We can then use these values to prune the dataset and train a model.

```py
seed=0
python value_clothing1M.py --seed=${seed} --cuda_num=0 --n_gpu=8 --slp \
    --prune_percs 0.1 0.2 0.3 0.4 --train_batch_size=512 --tag=slp_n10k_${seed} \
    --value_batch_size=512 --values_tag=supervised_prototypes_n10000_test_lr005_s${seed} \
    --wd=0.002
```