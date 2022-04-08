# JAX Implementation of bart-base

* [1. Motivation](#1-motivation)
* [2. Architecture](#2-architecture)
    * [2.1. Dropout function](#21-dropout-function)
    * [2.2. Layer Norm](#22-layer-norm)
    * [2.3. Embedding](#23-embedding)
    * [2.4. Linear](#24-linear)
    * [2.5. Attention](#25-attention)
    * [2.6. Transformer Encoder](#26-transformer-encoder)
    * [2.7. Transformer Decoder](#27-transformer-decoder)
    * [2.8. Transformer](#28-transformer)
* [3. Parameters](#3-parameters)
    * [3.1. Overview](#31-overview)
    * [3.2. Original checkpoint](#32-original-checkpoint)
    * [3.3. Flax BART model in Hugging Face](#33-flax-bart-model-in-hugging-face)
    * [3.4. This project](#34-this-project)
* [4. Training](#4-training)
* [5. Evaluation](#5-evaluation)
* [6. Implementation Notes](#6-implementation-notes)
    * [6.1. The bart-large model itself does not work properly](#61-the-bart-large-model-itself-does-not-work-properly)
    * [6.2. np.std and torch.std are different](#62-npstd-and-torchstd-are-different)
    * [6.3. Computations on TPU are in low precision by default](#63-computations-on-tpu-are-in-low-precision-by-default)
    * [6.4. BART has extra bias parameters for Layer Norm](#64-bart-has-extra-bias-parameters-for-layer-norm)
    * [6.5. BART has extra bias parameters for <em>Q</em>, <em>K</em> and <em>V</em>](#65-bart-has-extra-bias-parameters-for-q-k-and-v)
    * [6.6. Positional encoding is learned rather than fixed](#66-positional-encoding-is-learned-rather-than-fixed)
    * [6.7. Positional encoding has an offset of 2](#67-positional-encoding-has-an-offset-of-2)
    * [6.8. BART uses tied word embeddings](#68-bart-uses-tied-word-embeddings)
    * [6.9. BART has extra dropout after activation](#69-bart-has-extra-dropout-after-activation)
    * [6.10. Hugging Face Transformers 4.17.0 is not compactible with JAX 0.3.4](#610-hugging-face-transformers-4170-is-not-compactible-with-jax-034)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->

## 1. Motivation

This project is the JAX implementation of the [bart-base](https://arxiv.org/abs/1910.13461) model. It is built with two objectives in mind:

(1) To explain the BART architecture more clearly;

(2) To demonstrate how the Transformer-based model can be implemented in JAX.

In addition to the regular implementation, I also implemented the model [in a single line of Python code](https://twitter.com/ayaka14732/status/1507955631109869574), by virtue of JAX's functional-style API.

This project is inspired by [hyunwoongko/transformer](https://github.com/hyunwoongko/transformer). Nevertheless, the code is written entirely on my own.

## 2. Environment Setip

Setup on Cloud TPU

1\. Create a Cloud TPU VM v3-8 with TPU software version v2-nightly20210914

2\. Install Python 3.10

3\. Create virtualenv

4\. Install JAX with TPU support

```sh
pip install -U pip
pip install -U wheel
pip install "jax[tpu]==0.3.5" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

5\. Install TPU version of Tensorflow

```sh
wget https://gist.github.com/ayaka14732/a22234f394d60a28545f76cff23397c0/raw/e6c6ffea91b45a146189b52fea7155b1305bf78e/tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl.0
wget https://gist.github.com/ayaka14732/a22234f394d60a28545f76cff23397c0/raw/e6c6ffea91b45a146189b52fea7155b1305bf78e/tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl.1
cat tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl.0 tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl.1 > tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl
pip install tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl
rm -f tensorflow-2.8.0-cp310-cp310-linux_x86_64.whl*
```

6\. Install other required Python packages

```sh
pip install -r requirements.txt
```


## 2. Architecture

- Dropout function
- Layer Norm
- Embedding
- Linear
- Attention
- Transformer Encoder
- Transformer Decoder
- Transformer

## 3. Parameters

### 3.1. Overview

### 3.2. Original checkpoint

### 3.3. Flax BART model in Hugging Face

![](assets/parameter-format-1.svg)

```
shared
    embedding (50265, 768)
encoder
    embed_positions
        embedding (1026, 768)
    layernorm_embedding
        scale (768,)
        bias (768,)
    layers
        0..5
            self_attn
                q_proj
                    kernel (768, 768)
                    bias (768,)
                k_proj
                    kernel (768, 768)
                    bias (768,)
                v_proj
                    kernel (768, 768)
                    bias (768,)
                out_proj
                    kernel (768, 768)
                    bias (768,)
            self_attn_layer_norm
                scale (768,)
                bias (768,)
            fc1
                kernel (768, 3072)
                bias (3072,)
            fc2
                kernel (3072, 768)
                bias (768,)
            final_layer_norm
                scale (768,)
                bias (768,)
decoder
    embed_positions
        embedding (1026, 768)
    layernorm_embedding
        scale (768,)
        bias (768,)
    layers
        0..5
            self_attn
                q_proj
                    kernel (768, 768)
                    bias (768,)
                k_proj
                    kernel (768, 768)
                    bias (768,)
                v_proj
                    kernel (768, 768)
                    bias (768,)
                out_proj
                    kernel (768, 768)
                    bias (768,)
            self_attn_layer_norm
                scale (768,)
                bias (768,)
            encoder_attn
                q_proj
                    kernel (768, 768)
                    bias (768,)
                k_proj
                    kernel (768, 768)
                    bias (768,)
                v_proj
                    kernel (768, 768)
                    bias (768,)
                out_proj
                    kernel (768, 768)
                    bias (768,)
            encoder_attn_layer_norm
                scale (768,)
                bias (768,)
            fc1
                kernel (768, 3072)
                bias (3072,)
            fc2
                kernel (3072, 768)
                bias (768,)
            final_layer_norm
                scale (768,)
                bias (768,)
```

### 3.4. This project

![](assets/parameter-format-2.svg)

```
embedding
    embedding (50265, 768)
encoder_embed_positions (1026, 768)
decoder_embed_positions (1026, 768)
encoder_embed_layer_norm
    scale (768,)
    bias (768,)
decoder_embed_layer_norm
    scale (768,)
    bias (768,)
encoder_layers
    0..5
        self_attn
            q_proj
                kernel (12, 768, 64)
                bias (12, 64)
            k_proj
                kernel (12, 768, 64)
                bias (12, 64)
            v_proj
                kernel (12, 768, 64)
                bias (12, 64)
            ff
                kernel (768, 768)
                bias (768,)
        self_attn_layer_norm
            scale (768,)
            bias (768,)
        ff0
            kernel (768, 3072)
            bias (3072,)
        ff1
            kernel (3072, 768)
            bias (768,)
        final_layer_norm
            scale (768,)
            bias (768,)
decoder_layers
    0..5
        self_attn
            q_proj
                kernel (12, 768, 64)
                bias (12, 64)
            k_proj
                kernel (12, 768, 64)
                bias (12, 64)
            v_proj
                kernel (12, 768, 64)
                bias (12, 64)
            ff
                kernel (768, 768)
                bias (768,)
        self_attn_layer_norm
            scale (768,)
            bias (768,)
        cross_attn
            q_proj
                kernel (12, 768, 64)
                bias (12, 64)
            k_proj
                kernel (12, 768, 64)
                bias (12, 64)
            v_proj
                kernel (12, 768, 64)
                bias (12, 64)
            ff
                kernel (768, 768)
                bias (768,)
        cross_attn_layer_norm
            scale (768,)
            bias (768,)
        ff0
            kernel (768, 3072)
            bias (3072,)
        ff1
            kernel (3072, 768)
            bias (768,)
        final_layer_norm
            scale (768,)
            bias (768,)
```

## 4. Training

## 5. Evaluation

## 6. Implementation Notes

This section records the problems I encountered during my implementation of the BART model and the final solutions.

### 6.1. The bart-large model itself does not work properly

This issue is reported in [huggingface/transformers#15559](https://github.com/huggingface/transformers/issues/15559). As a consequence, I only focus on implementing bart-base in this project, and not bart-large.

### 6.2. `np.std` and `torch.std` are different

```python
import torch

x = torch.tensor([[-1., 1.]])

print(x.std(-1).numpy())  # [1.4142135]
print(x.numpy().std(-1))  # [1.]
```

It is because in `np.std` the denominator is _n_, while in `torch.std` it is _n_-1. See [pytorch/pytorch#1854](https://github.com/pytorch/pytorch/issues/1854) for details.

However, for the standard deviation in Layer Norm, the denominator is always n in either PyTorch or NumPy.

### 6.3. Computations on TPU are in low precision by default

JAX uses bfloat16 for matrix multiplication on TPU by default, even if the data type is float32. See [google/jax#9973](https://github.com/google/jax/issues/9973) for details.

```python
import jax.numpy as np

print(4176 * 5996)  # 25039296

a = np.array(0.4176, dtype=np.float32)
b = np.array(0.5996, dtype=np.float32)
print((a * b).item())  # 0.25039297342300415
```

For neural network training, however, reducing the accuracy is worthwhile because it can significantly reduce the training time, according to Tom's comments in the above issue.

### 6.4. BART has extra bias parameters for Layer Norm

In section 2.1 of the BART paper, it is stated that BART uses the standard Transformer architecture, except for the activation function and initialization. However, this is not true because BART has extra bias parameters for Layer Norm.

TODO: Add the formula of Layer Norm here.

TODO: Add a proof that the original Transformer architecture does not have bias for Layer Norm.

### 6.5. BART has extra bias parameters for _Q_, _K_ and _V_

Besides Layer Norm, BART also has has extra bias parameters for _Q_, _K_ and _V_.

TODO: Add demonstration.

### 6.6. Positional encoding is learned rather than fixed

### 6.7. Positional encoding has an offset of 2

### 6.8. BART uses tied word embeddings

### 6.9. BART has extra dropout after activation

### 6.10. Hugging Face Transformers 4.17.0 is not compactible with JAX 0.3.4
