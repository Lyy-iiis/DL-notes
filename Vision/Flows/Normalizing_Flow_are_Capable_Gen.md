# Normalizing Flows are Capable Generative Models

## Preliminaries

$$
p_{model}(x)=p_0(f(x))\left|\det\left(\frac{\partial f(x)}{\partial x}\right)\right|
$$
Maximum likelihood estimation (MLE) objective
$$
\min_f -\log p_0(f(x))-\log\left|\det\left(\frac{\partial f(x)}{\partial x}\right)\right|=\min_f 0.5\|f(x)\|^2-\log\left|\det\left(\frac{\partial f(x)}{\partial x}\right)\right|
$$

## Block Autoregressive Flows

Input $x\in\mathbb{R}^{N\times D}$, number of flow layers $T$
$$
z^T=f(x)=(f^{T-1}\circ\cdots\circ f^0)(x)
$$
the $t$-th flow is parameterized by two function $\mu^t,\alpha^t:\mathbb{R}^{N\times D}\to \mathbb{R}^{N\times D}$, transform $z^t$ to $z^{t+1}$ as
$$
\tilde{z}^t=\pi^t(z^t),\quad z^{t+1}_i=(\tilde{z}^t_i-\mu^t_i(\tilde{z}^t))\odot\exp(-\alpha^t_i(\tilde{z}^t_{<i}))
$$
and reverse transformation
$$
\tilde{z}^t_i=z^{t+1}_i\odot\exp(\alpha^t_i(\tilde{z}^t_{<i}))+\mu^t_i(\tilde{z}_{<i}^t)\quad z^t=(\pi^t)^{-1}(\tilde{z}^t)
$$
where $\pi^t$ is a permutation of the input dimensions, the loss function is
$$
\min_f 0.5\|z^T\|^2+\sum_{t=0}^{T-1}\sum_{i=1}^{N-1}\sum_{j=0}^{D-1}\alpha^t_{i}(\tilde{z}^t_{<i})_{j}
$$
where $\alpha$ and $\mu$ is implemented by ViT.

### Noise Augmented Training

Train: input $x$ by probability $\tilde{p}(\tilde{x})=\int_{\epsilon\in [0,bin]^D}p_{model}(\tilde{x}+\epsilon)d\epsilon$
Inference: set $p_\epsilon$ as Gaussian distribution $\mathcal{N}(0,\sigma^2I)$ whose $\sigma$ is around 0.05 where pixel values are in $[-1,1]$

Adding noise is necessary for generalization, the author explain it clearly in the paper

However, our train model is trained on noisy image, so we need to denoise
joint distribution $(x,y)$ where $x\sim p_{data},y=x+\epsilon$ for $\epsilon\sim\mathcal{N}(0,\sigma^2I)$
$$
x=\mathbb{E}_{\epsilon\sim\mathcal{N}(0,\sigma^2I)}[x|y]=y+\sigma^2\nabla_y\log q(y)=y+\sigma^2\nabla_y\log p_{model}(y)
$$

### Guidance

Conditional Guidance: $\mu_i^t(\cdot;c),\alpha_i^t(\cdot;c)$ where $c$ is the class, and $\mu_i^t(\cdot;\emptyset),\alpha_i^t(\cdot;\emptyset)$ is trained by mask class randomly. When inference
$$
\tilde{z}^t_i=z^{t+1}_i\odot\exp(\alpha^t_i(\tilde{z}^t_{<i};c,w))+\mu^t_i(\tilde{z}_{<i}^t;c,w)
$$
where
$$
\alpha^t_i(\tilde{z}^t_{<i};c,w)=(1+w)\alpha^t_i(\tilde{z}^t_{<i};c)-w\alpha^t_i(\tilde{z}^t_{<i};\emptyset)\\
\mu^t_i(\tilde{z}_{<i}^t;c,w)=(1+w)\mu^t_i(\tilde{z}_{<i}^t;c)-w\mu^t_i(\tilde{z}_{<i}^t;\emptyset)
$$
where $w$ is the weight of guidance

Unconditional Guidance: $\mu_i^t(\cdot;\tau),\alpha_i^t(\cdot;\tau)$ where $\tau$ is the temperature by divide attention score logits by $\tau$
$$
\alpha^t_i(\tilde{z}^t_{<i};\tau,w)=(1+w)\alpha^t_i(\tilde{z}^t_{<i};1)-w\alpha^t_i(\tilde{z}^t_{<i};\tau)\\
\mu^t_i(\tilde{z}_{<i}^t;\tau,w)=(1+w)\mu^t_i(\tilde{z}_{<i}^t;1)-w\mu^t_i(\tilde{z}_{<i}^t;\tau)
$$

Here is our reproduce: [code](https://github.com/Lyy-iiis/Normalizing_Flow/tree/tarflow)