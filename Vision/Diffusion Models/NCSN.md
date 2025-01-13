# Noise Conditional Score Networks

## Background

### Score-Based Generative Modeling

Our goal is use $s_\theta(x)$ to estimate $\nabla_x \log p_{\text{data}}(x)$, so

$$
\arg\min_\theta \frac{1}{2}\mathbb{E}_{x\sim p_{\text{data}}(x)}\left[\|s_\theta(x)-\nabla_x \log p_{\text{data}}(x)\|^2\right]
$$

i.e.

$$
\arg\min_\theta \mathbb{E}_{x\sim p_{\text{data}}(x)}\left[Tr(\nabla_x s_\theta(x))+\frac{1}{2}\|s_\theta(x)\|^2\right]
$$

However, $Tr(\nabla_x s_\theta(x))$ is hard to optimize. Sol:

#### Denosing Score Matching

Define $q_\sigma(\tilde{x})=\int q_\sigma(\tilde{x}|x)p_{\text{data}}(x)dx$, then

$$
\frac{1}{2}\mathbb{E}_{\tilde{x}\sim q_\sigma(\tilde{x})}\left[\|s_\theta(\tilde{x})-\nabla_{\tilde{x}} \log q_\sigma(\tilde{x})\|^2\right]\\
=\frac{1}{2}\mathbb{E}_{\tilde{x}\sim q_\sigma(\tilde{x})}\left[\|s_\theta(\tilde{x})\|^2\right]-\mathbb{E}_{x\sim p_{\text{data}}(x),\tilde{x}\sim q_\sigma(\tilde{x}|x)}\left[\nabla_{\tilde{x}}\log q_\sigma(\tilde{x})\cdot s_\theta(\tilde{x})\right]+C\\
=\frac{1}{2}\mathbb{E}_{\tilde{x}\sim q_\sigma(\tilde{x})}\left[\|s_\theta(\tilde{x})\|^2\right]-\iint \nabla_{\tilde{x}} \left(q_\sigma(\tilde{x}|x)p_{\text{data}}(x)\right) s_\theta(\tilde{x})d\tilde{x}dx+C\\
=\frac{1}{2}\mathbb{E}_{\tilde{x}\sim q_\sigma(\tilde{x})}\left[\|s_\theta(\tilde{x})\|^2\right]-\iint q_\sigma(\tilde{x}|x)p_{\text{data}}(x)\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x)\cdot s_\theta(\tilde{x})d\tilde{x}dx+C\\
=\frac{1}{2}\mathbb{E}_{x\sim p_{\text{data}}(x),\tilde{x}\sim q_\sigma(\tilde{x}|x)}\left[\|s_\theta(\tilde{x})-\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x)\|^2\right]+C
% =\frac{1}{2}\mathbb{E}_{x\sim q_\sigma(x)}\left[\|s_\theta(x)\|\right]^2-\int q_\sigma(x)\nabla_x\log q_\sigma(x)\cdot s_\theta(x)dx+C\\
% =\frac{1}{2}\mathbb{E}_{x\sim q_\sigma(x)}\left[\|s_\theta(x)\|\right]^2-\mathbb{E}_{x\sim p_{\text{data}}(x),\tilde{x}\sim q_\sigma(\tilde{x}|x)}\left[\nabla_x\log q_\sigma(\tilde{x}|x)\cdot s_\theta(x)\right]-\mathbb{E}_{x\sim p_{\text{data}}(x)}[\nabla_x p_{\text{data}}(x)\cdot s_\theta(x)]+C
$$

we can show optimal score network satisfies $s_\theta(x)=\nabla_x \log q_\sigma(x)$ almost surely. But only when $\sigma\to 0$, $q_\sigma(x)\to p_{\text{data}}(x)$, however, assuming Gaussian

$$
\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x)=\frac{\tilde{x}-x}{\sigma^2}\\
\frac{1}{2}\mathbb{E}_{\tilde{x}\sim q_\sigma(\tilde{x})}\left[\|s_\theta(\tilde{x})-\nabla_{\tilde{x}} \log q_\sigma(\tilde{x})\|^2\right]=\frac{1}{2}\mathbb{E}_{x\sim p_{\text{data}}(x),z\sim N(0,I)}\left[\|s_\theta(x+\sigma z)+\frac{z}{\sigma}\|^2\right]
$$

which has infinite variance when $\sigma\to 0$.

#### Sliced Score Matching

Use random projection to approx $Tr(\nabla_x s_\theta(x))$, objective is

$$
\mathbb{E}_{v\sim p_v}\mathbf{E}_{x\sim p_{\text{data}}(x)}\left[v^T\nabla_x s_\theta(x)v+\frac{1}{2}\|s_\theta(x)\|^2\right]
$$

### Sampling

Now suppose we have $\nabla_x\log p(x)$,  how can we sample from $p(x)$? __Langevin dynamics__

We can prove using

$$
x_t=x_{t-1}+\frac{\epsilon}{2}\nabla_x\log p(x_{t-1})+\sqrt{\epsilon}\eta_t
$$

where $\eta_t\sim N(0,I)$, then $x_T\sim p(x)$ when $T\to\infty,\epsilon\to 0$ under some conditions.

#### Problems

1. Data in real world concentrates on low dim manifold, then $\nabla_x\log p(x)$ is not well defined.
2. Can't estimate well when $p(x)$ is extremely low, which appears almost everywhere in real world.
3. For $p_1(x),p_2(x)$ with different support, $p(x)=\pi p_1(x)+(1-\pi)p_2(x)$, then $\nabla_x\log p(x)$ doesn't corelate with $\pi$ -> Annealed Langevin dynamics

## Learning and Inference

### Intuition

1. If using a Gaussian noise to perturb our data, manifold of data will be expanded
2. Gaussian noise with high $\sigma$ will solve low density problem
3. A sequence of Gaussian noise with decreasing $\sigma$ will help us to estimate $\pi$

### Learning

Learn $s_\theta(x,\sigma)=\nabla_x\log q_\sigma(x)$, where $q_\sigma(x)=\int p_{\text{data}}(x')N(x|x',\sigma^2I)dx'$

$$
\ell(\theta;\sigma)=\frac{1}{2}\mathbb{E}_{x\sim p_{\text{data}}(x),\tilde{x}\sim q_\sigma(\tilde{x}|x)}\left[\|s_\theta(\tilde{x},\sigma)+\frac{\tilde{x}-x}{\sigma^2}\|^2\right]
% =\mathbb{E}_{x\sim p_{\text{data}}(x)}\left[\|s_\theta(x,\sigma)-\nabla_x\log q_\sigma(x)\|^2\right]\\
$$

our loss is defined as

$$
\ell(\theta)=\frac{1}{L}\sum_{i=1}^L\lambda(\sigma_i)\ell(\theta;\sigma_i)
$$

to make each $\sigma_i$ has equal contribution, we can set $\lambda(\sigma)=\sigma^2$

i.e.

$$
\ell(\theta)=\frac{1}{L}\sum_{i=1}^L\mathbb{E}_{x\sim p_{\text{data}}(x),z\sim N(0,I)}\left[\|\sigma_i s_\theta(x+\sigma_i z,\sigma_i)+z\|^2\right]
$$

### Inference

Define $\alpha_i=\epsilon \frac{\sigma_i^2}{\sigma_L^2}$, then denoise $T$ steps using step size $\alpha_i$, from $i=L$ to $1$

Using Annealed Langevin dynamics, the sampling process can reflect $\pi$