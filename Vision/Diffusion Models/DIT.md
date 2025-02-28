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

## SiT: Scalable Interpolnt Transformers

TLDR: Apply flow matching to DiT, explore __design space__ for SiT

Denote $x_t=\alpha_t x_*+\sigma_t\epsilon$

Probability flow ODE:
$$
v(x,t)=\mathbb{E}[\dot{x}_t|x_t=x]=\dot{\alpha}_t\mathbb{E}[x_*|x_t=x]+\dot{\sigma}_t\mathbb{E}[\epsilon|x_t=x]
$$
Reverse-time SDE:
$$
dX_t=v(X_t,t)dt-\frac{1}{2}w_ts(X_t,t)dt+\sqrt{w_t}dW_t
$$
where $s(x,t)=\nabla\log p_t(x)$ is the score, given by
$$
s(x,t)=-\sigma_t^{-1}\mathbb{E}[\epsilon|x_t=x]
$$

We can choose to estimate score or velocity using
- $\mathcal{L}_S(\theta)=\int_0^T\mathbb{E}[||\sigma_t s_\theta(x_t,t)+\epsilon||^2]dt$
- $\mathcal{L}_V(\theta)=\int_0^T\mathbb{E}[||v_\theta(x_t,t)-(\dot{\alpha}_t x_*+\dot{\sigma_t}\epsilon)||^2]dt$

and they are related by
$$
s(x,t)=\sigma_t^{-1}\frac{\alpha_t v(x,t)-\dot{\alpha}_t x}{\dot{\alpha}_t\sigma_t-\alpha_t\dot{\sigma}_t}
$$

### Design space
- What is the prediction target of the model? $v(x,t)$ or $s(x,t)$
- The choice of time discretization, continuous (flow) or discrete (diffusion)
- The choice of interpolant ($\alpha_t,\sigma_t$)
  - Score-based: VP SDE $dX_t=-\frac{1}{2}\beta_t X_tdt+\sqrt{\beta_t}dW_t$, then $\alpha_t=e^{-\frac{1}{2}\int_0^t \beta_sds},\sigma_t=\sqrt{1-e^{-\int_0^t \beta_sds}}$
  - General interpolant: w/o any explicit forward SDE
- The diffusion coefficient $w_t$ in the reverse-time SDE, only in inference time

### Experiments

- Predict $v$ is better than $s$, but by selecting $\lambda(t)$ such that weighted score matching is equivalent to velocity matching, we can achieve similar performance.
- Continuous-time flow matching is better than discrete-time diffusion matching.
- Linear ($\alpha_t=1-t,\sigma_t=t$) and GVP ($\alpha_t=\cos(\pi t/2),\sigma_t=\sin(\pi t/2)$) interpolants are better than SBDM-VP.
- SDE is better than ODE, with $w_t=w_t^{KL}$ or $w_t=w_t^{KL,\eta}$