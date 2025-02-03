# Scalable Diffusion Models with Transformers

## Motivation

Most diffusion models are based on U-Net as backbone. 
- Is inductive bias of U-Net necessary for diffusion models? 
- Can we replace U-Net with Transformers, in order to scale up diffusion models?

Before replacing U-Net in diffusion, what is the advantage of U-net ?
- The input size and output size of U-net is same, with residue connnection, which is suitable for Diffusion naturally.
- In transformer, it seems too hard for model to predict $\epsilon$ or $x_{t-1}$ from $x_t$. (I don't know why I feel this way, but directly generate a image like thing from a image makes me feel uncomfortable)
- **Diffusion in latent space is more natural for transformer**

## Method

Main idea:
1. Learn an autoencoder with encoder $E$ and decoder $D$ that compresses images into smaller spatial representations.
2. Train diffusion models on $z=E(x)$, then decode $z$ back to $x$ when inference using $D$.

Design:
- Patchify: the input to DiT is a spatial representation $I\times I\times C$, with patch size $p\times p$, convert input into $(I/p)\times (I/p)$ tokens, dim of each token is $p\times p\times C$.
- DiT block design: treat vector embeddings of $t$ and $c$ as additional tokens. 3 kinds of DiT blocks design: adaLN-zero, cross-attention, in-context conditioning
![DiT](fig/DiT.png)

Experiment results:
- Smaller patch size, larger transformer size works better.
- DiT Gflops are critical to improving performance, higher Gflops, better performance.
- Larger DiT models are more compute-efficient due to smaller training steps.