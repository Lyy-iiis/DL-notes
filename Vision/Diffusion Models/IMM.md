# Inductive Moment Matching

## Preliminaries

Maximum Mean Discrepancy (MMD) between distributions $p$ and $q$ defined on Reproducing Kernel Hilbert Spaces (RKHS) $\mathcal{H}$ with kernels $k$: $\mathbb{R}^d \times \mathbb{R}^d \rightarrow \mathbb{R}$ is defined as:

$$
\text{MMD}^2(p, q) = \left\| \mathbb{E}_{x \sim p}[\phi(x)] - \mathbb{E}_{x \sim q}[\phi(x)] \right\|_{\mathcal{H}}^2
$$

where $\phi$ is the feature map associated with the kernel $k$. Choices such as RBF kernel imply an inner product of infinite-dimensional feature maps.

Notation: $q_t(x_t|x,\epsilon)=\mathcal{N}(I_t(x,\epsilon), \gamma_t^2 I)$ with constraints $I_1(x,\epsilon)=\epsilon$ and $I_0(x,\epsilon)=x$, $\gamma_0=\gamma_1=0$.

## Method

$$
q_t(x_t)=\iint q_t(x_t|x,\epsilon)q(x)p(\epsilon)dx d\epsilon
$$

We want to learn a one-step sampler that transforms $q_t(x_t)$ into $q_s(x_s)$ for some $s\leq t$. We define $x_s$ as a generalized interpolant between $x$ and $x_t$ if for all $s\in [0,t]$, its distribution follows

$$
q_{s|t}(x_s|x,x_t)=\mathcal{N}(I_{s|t}(x,x_t),\gamma_{s|t}^2 I)
$$

with constraints $I_{t|t}(x,x_t)=x_t,I_{0|t}(x,x_t)=x,\gamma_{t|t}=\gamma_{0|t}=0$ and $q_{t|1}(x_t|x,\epsilon)=q_t(x_t|x,\epsilon)$

> Def: Marginal-Perserving Interpolants if
>
> $$
> q_s(x_s)=\iint q_{s|t}(x_s|x,x_t)q_t(x|x_t)q_t(x_t)dx_tdx
> $$
>
> where
>
> $$
> q_t(x|x_t)=\int \frac{q_t(x_t|x,\epsilon)q(x)p(\epsilon)}{q_t(x_t)}d\epsilon
> $$

We hope our model is marginal-perserving with one step sampler $p_{s|t}^\theta(x|x_t)$, i.e.

$$
p^\theta_{s|t}(x_s)=\iint q_{s|t}(x_s|x,x_t)p_{s|t}^\theta(x|x_t)q_t(x_t)dx_tdx
$$

is same as $q_s(x_s)$, we can minimizing

$$
\mathcal{L}(\theta)=\mathbb{E}_{s,t}[D(q_s(x_s),p^\theta_{s|t}(x_s))]
$$

> note: why $p_{s|t}^\theta(x|x_t)$ involves $s$ ? we can imagine we are done this by computing $x_s$, then get $x$ by interpolating.

### Learning via Inductive Bootstrapping

It's difficult to optimize $\mathcal{L}(\theta)$ directly. We should use bootstrap, i.e. increase the gap between $s$ and $t$ step by step (find $s\leq r(s,t)\leq t$)
$$
\mathcal{L}(\theta_n)=\mathbb{E}_{s,t}[w(s,t)\text{MMD}^2(p^{\theta_{n-1}}_{s|r}(x_s),p^{\theta_n}_{s|t}(x_s))]
$$
**We can prove optimizing this objective results in target distribution $q_s(x_s)$.**

## Simplified Formulation

> Def Self-Consistent Interpolants: 
> $$
> q_{s|t}(x_s|x,x_t)=\int q_{s|r}(x_s|x,x_r)q_{r|t}(x_r|x_t)dx_r
> $$

A self-consistent interpolant is DDIM-interpolant, i.e.
$$
\gamma_{s|t}=0\quad I_{s|t}(x,x_t)=DDIM(x_t,x,s,t)=(\alpha_s-\frac{\sigma_s}{\sigma_t}\alpha_t)x+\frac{\sigma_s}{\sigma_t}x_t
$$
A deterministic $p_{s|t}^\theta(x|x_t)$ is sufficient to achieve zero MMD. 
We can define $p_{s|t}^\theta(x|x_t)=\delta_{x-g_\theta(x_t,s,t)}$, then sample $x_s$ by $x_s=DDIM(x_t,g_\theta(x_t,s,t),s,t)$.

Final objective
$$
\mathcal{L}_{IMM}(\theta)=\mathbb{E}_{x_t,x_t',x_r,x_r',s,t}[w(s,t)[k(y_{s,t},y_{s,t}')+k(y_{s,r},y_{s,r}')-k(y_{s,t},y_{s,r}')-k(y_{s,t}',y_{s,r})]]
$$
where $y_{s,t}=f_{s,t}^\theta(x_t),y_{s,t}'=f_{s,t}^\theta(x_t'),y_{s,r}=f_{s,r}^{\theta^-}(x_r),y_{s,r}'=f_{s,r}^{\theta^-}(x_r')$ and $k$ is a kernel function. $x_r,x_r'$ are obtained by reusing $x_t,x_t'$.

Implementation details ignored here :)