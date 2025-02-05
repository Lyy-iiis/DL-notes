# Masked Autoencoders Are Scalable Vision Learners

## Motivation

BERT use masked autoencoding to learn features: remove a portion of data and learn to predict removed content.

What makes masked autoencoding different between vision and language?
- Conv nets operate on regular grid, not straightforward to mask or add position embeddings. 
  - Sol: ViT
- Information density of language and vision is different. A little high-level understanding of image can be enough to recover missing content. 
  - Sol: Mask high portion of random patches to make task more challenging.
- Autoencoder's decoder needs to maps the latent representation back to pixel, which is high level to low level mapping. 
  - Sol: Careful design of decoder.

## Method

MAE use asymmetric design of encoder and decoder. 

1. Masking: divide image into non-overlapping patches, sample a subset from patches
2. Encoder: linear projection of patches, add positional embeddings, remove masked patches, apply transformer encoders.
3. Decoder: combine encoded visible patches and mask tokens(one shared embedding) together, add positional embeddings, apply transformer decoders (smaller, narrower, and shallower than encoder).
4. Reconstruction: linear projection of decoder output to pixel space, compute MSE loss.

The method is really simple and elegant, but effective and scalable !