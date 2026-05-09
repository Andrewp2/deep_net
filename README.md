# deep_net

Experiment for training absurdly deep residual networks on MNIST. The current
main target is a 10-million-layer dense width-4 residual network whose counted
core does the actual classification work.

Each counted dense layer is:

```text
h = h + alpha * activation(h W_i + b_i)
```

For the current run:

```text
layers:     10,000,000
width:      4
activation: softsign
batch:      16
optimizer:  exact SGD
weights:    0.8 GB
```

The width-4 softsign path uses a hand-written AVX2 batch-lane kernel and can
load the full weight file into RAM for training.

## Quick smoke test

```sh
cargo run --release -- train \
  --layers 10000 \
  --weights /tmp/deep_net_smoke_weights.bin \
  --head /tmp/deep_net_smoke_head.bin \
  --steps 1 \
  --batch 2 \
  --chunk-layers 1000 \
  --train-limit 128
```

## Main Run

This is the run to try first. It trains a 10M-layer width-4 residual core for
1000 minibatches, then writes the weights back to disk:

```sh
cargo run --release -- train-dense \
  --layers 10000000 \
  --width 4 \
  --weights /media/andrew-peterson/HardDrive/deep_net/weights_10m_w4_softsign.bin \
  --head /media/andrew-peterson/HardDrive/deep_net/head_10m_w4_softsign.bin \
  --data-dir data/mnist \
  --steps 1000 \
  --batch 16 \
  --chunk-layers 100000 \
  --activation softsign \
  --layer-lr 0.01 \
  --head-lr 0.01 \
  --report-every 10 \
  --in-memory
```

Then evaluate on the full 10,000-image MNIST test set:

```sh
cargo run --release -- eval-dense \
  --layers 10000000 \
  --width 4 \
  --weights /media/andrew-peterson/HardDrive/deep_net/weights_10m_w4_softsign.bin \
  --head /media/andrew-peterson/HardDrive/deep_net/head_10m_w4_softsign.bin \
  --data-dir data/mnist \
  --samples 10000 \
  --batch 256 \
  --chunk-layers 100000 \
  --activation softsign
```

Expected runtime on this machine:

```text
step time:        ~1.0-1.2s
1000-step train:  ~17-20 min of step time
RAM use:          ~1 GB resident
```

Completed 1000-step result:

```text
final train minibatch loss: 0.846
test loss:                  0.965448
test acc:                   0.701
test eval time:             69.17s
```

## Width-2 100M Run

Put the large weight file on the mounted hard drive:

```sh
cargo run --release -- train \
  --layers 100000000 \
  --weights /media/andrew-peterson/HardDrive/deep_net/weights_100m.bin \
  --head /media/andrew-peterson/HardDrive/deep_net/head_100m.bin \
  --steps 1 \
  --batch 1 \
  --chunk-layers 1000000 \
  --train-limit 60000
```

The first `train` run initializes the weight file if it is missing. To create it
explicitly:

```sh
cargo run --release -- init \
  --layers 100000000 \
  --weights /media/andrew-peterson/HardDrive/deep_net/weights_100m.bin \
  --chunk-layers 1000000
```

## Notes

This is exact reverse-mode SGD through the deep stack, but chunked:

- forward pass streams the layer file and saves chunk boundary activations
- backward pass streams chunks in reverse
- each reverse chunk recomputes local activations, applies SGD, and writes back

Width 2 is a severe bottleneck for MNIST. It can still train in the sense that
the full model is differentiable and updates all layers, but it should be
treated as a systems stunt rather than a good classifier.

## Good MNIST Loss

The pure width-2 path above is intentionally absurd and should not be expected
to get strong MNIST loss. For a run that actually learns MNIST, use the MLP head:

```sh
cargo run --release -- train-mlp \
  --head /media/andrew-peterson/HardDrive/deep_net/mlp_head_128.bin \
  --hidden 128 \
  --steps 1000 \
  --batch 128 \
  --lr 0.05

cargo run --release -- train-mlp \
  --head /media/andrew-peterson/HardDrive/deep_net/mlp_head_128.bin \
  --hidden 128 \
  --steps 4000 \
  --batch 128 \
  --lr 0.03

cargo run --release -- eval-mlp \
  --head /media/andrew-peterson/HardDrive/deep_net/mlp_head_128.bin \
  --hidden 128 \
  --samples 10000 \
  --batch 256
```

On this machine that reached:

```text
test loss: 0.174178
test acc:  0.951
```

This is the practical-good-loss path, not the pure streamed width-2 core. The
honest next step is to splice the good MNIST head into the deep experiment as a
residual/identity branch, or to widen the counted core beyond 2.

## Dense Core Notes

For the version where the counted layers do the real work, use the dense core:

```text
h = h + alpha * activation(h W_i + b_i)
```

where `h` has configurable width and each counted layer has a full dense
`width x width` matrix plus bias. Use the 10M-layer width-4 command in
`Main Run` for the current depth stunt; the shorter command below is just a
quick dense-core learning check.

```sh
cargo run --release -- train-dense \
  --layers 10000 \
  --width 32 \
  --weights /tmp/deep_net_dense_10k_w32.bin \
  --head /tmp/deep_net_dense_10k_w32_head.bin \
  --steps 1000 \
  --batch 1 \
  --chunk-layers 1000 \
  --layer-lr 0.01 \
  --head-lr 0.01 \
  --report-every 200
```

For the long width-4 depth stunt, `softsign` is the current default. It keeps
the same residual dense-layer shape but avoids billions of expensive `tanh`
calls.

Calibration on this machine:

```text
10M width-4 tanh step:     28.46s with native release build
10M width-4 softsign step: 10.76s with native release build
10M width-4 ReLU step:     11.11s with native release build
10M width-4 softsign step:  5.3s with native release + width-4 unrolled kernel
10M width-4 softsign step:  1.9-2.1s with AVX2 batch-lane kernel
10M width-4 softsign step:  1.0-1.2s with AVX2 + in-memory weights
```

On the 100-layer width-4 learning sweep:

```text
tanh:     loss 0.963126, acc 0.704
softsign: loss 1.007459, acc 0.694
ReLU:     loss 1.189358, acc 0.651
```

ReLU is almost as fast as softsign, but it learned worse in the quick width-4
sweep. Softsign is the better default for the long run unless a longer ReLU
sweep proves otherwise.

The width-4 softsign path has a hand-unrolled AVX2 kernel over the batch
dimension. It processes 8 MNIST samples per vector lane group and falls back to
the scalar unrolled kernel for non-AVX2 machines or non-softsign activations.

`--in-memory` loads the full dense weight file into RAM, trains there, and
writes it back once at the end. For 10M width-4 this uses about 1 GB resident
memory and avoids per-step file writeback.

A 10k-layer width-32 smoke run reached:

```text
1000-sample test loss: 0.744920
1000-sample test acc:  0.765
```

after 1200 online updates. That is not yet good MNIST, but it is already far
past the width-2 bottleneck and shows the dense counted core can learn.

Approximate dense storage at 100M layers:

```text
width 8:    28.8 GB
width 16:  108.8 GB
width 32:  422.4 GB
width 64: 1664.0 GB
```

Width 64 does not fit the current 1.3 TB free-space budget. Width 32 fits, but
one exact SGD step has a disk I/O floor around 1.27 TB, so the practical next
targets are width 32 at 1M-10M layers, or width 16 at higher depth.
