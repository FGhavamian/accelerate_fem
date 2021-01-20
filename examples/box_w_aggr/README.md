`python train.py \
--learning-rate 1e-3 \
--filter-sizes '[4,8,16,32]' \
--kernel-sizes '[[15,15],[19,19],[21,21]]' \
--epochs 50 \
--model-name v0 \
--continue-training 0 \
--initial-epoch 0`

`python train.py \
--learning-rate 1e-3 \
--filter-sizes '[4,8,16,32]' \
--kernel-sizes '[[15,15],[19,19],[21,21]]' \
--epochs 100 \
--model-name v0 \
--continue-training 1 \
--initial-epoch 50`