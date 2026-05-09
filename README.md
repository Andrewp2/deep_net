# deep_net

Disk-backed experiment for training an absurdly deep width-2 residual network on
MNIST.

Each counted layer is:

```text
h = h + alpha * tanh(h W_i + b_i)
```

with `h` width 2 and six `f32` parameters per layer. A 100M-layer file is
2.4 GB.

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

## 100M-layer run

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

## Dense Width 32/64 Core

For the version where the counted layers do the real work, use the dense core:

```text
h = h + alpha * tanh(h W_i + b_i)
```

where `h` has configurable width and each counted layer has a full dense
`width x width` matrix plus bias.

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
