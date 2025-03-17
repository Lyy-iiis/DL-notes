# Reconstruction vs. Generation: Taming Optimization Dilemma in Latent Diffusion Models

TLDR: change loss, achieve higher performance

## Align VAE with Vision Foundation models

Marginal Cosine Similarity Loss: 
Given image $I$, processed by encoder of visual tokenizer $E$ and frozen vision foundation model $f$, resulting in image latents $Z=E(I)$ and visual representation $F=f(I)$. Project to same dim $Z'=WZ$, the define 
$$
\mathcal{L}_{mcos}=\frac{1}{hw}\sum_{i,j}ReLU(1-m_1-\frac{F_{ij}^T Z'_{ij}}{\|F_{ij}\|\|Z'_{ij}\|})
$$
where $m_1$ is a margin, $i,j$ are pixel indices.

Marginal Distance Matrix Similarity Loss:
Align internal distributions of feature matrices $z$ and $f$
$$
\mathcal{L}_{mdms}=\frac{1}{N^2}\sum_{i,j}ReLU\left(|\frac{z_i^Tz_j}{\|z_i\|\|z_j\|}-\frac{f_i^Tf_j}{\|f_i\|\|f_j\|}|-m_2\right)
$$
where $m_2$ is a margin.

Adaptive Weighting: define 
$$w_{adapt}=\frac{\|\nabla L_{rec}\|}{\|\nabla L_{mcos}+\nabla L_{mdms}\|}$$
and $\mathcal{L}_{vf}=w_{hyper}w_{adapt}(L_{mcos}+L_{mdms})$

## Experiments

Observation: higher latent dimensionality leads to better reconstruction but worse generation quality.