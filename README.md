# E.T.
## Intro
This repository is associated with the paper published in SC21.
> E.T. Re-Thinking Self-Attention for Transformer Models on GPUs

It contains some implemented kernels mentioned in the paper and a few examples of encoder.

## Platform

Tested on NVIDIA V100S GPU with CUDA 11.4.

## Example

There are three examples of encoders in ```test```, all of which use random data.

1. On-the-fly attention with tensor-tile pruned linear transformations (```encoder_tile_test```)
2. Attention-aware pruning with pruned self-attention (```encoder_prune_test```)
3. Sequence-aware optimized encoder (```encoder_length_test```)

## build 
```
mkdir build && cd build 
cmake .. 
make -j
```


