# Consistency Models

## Background 

Diffusion models suffer from the problem of iterative sampling. 
We want single-step generation, which is sufficient to learn a model maps $x$ and $t$ to $x_0$.

Property: points on the same trajectory map to same initial point.

Diffusion models start by diffusing $p_{\text{data}}(x)$ with SDE

$$
dx_t = \mu(x_t, t) dt + \sigma(t) dw_t
$$

where $w_t$ is Brownian motion, the corresponding Fokker-Planck equation is

$$
\frac{\partial p(x, t)}{\partial t} = -\nabla \cdot (\mu(x, t) p(x, t)) + \frac{1}{2} \nabla^2 p(x, t)
$$

hence we can use a ODE to get same $p(x, t)$

$$
d x_t = \left(\mu(x_t, t)-\frac{1}{2} \sigma^2(t) \nabla \log p(x_t, t)\right) dt 
$$

## Method 

### Consistency Models

Given solution trajectory $\{x_t\}_{t\in [\epsilon,T]}$ of ODE, we define consistency function as $f: (x_t,t)\to x_\epsilon$ which satisfies

$$
f(x_t, t) = f(x_{t'}, t') \quad \text{for all } t, t' \in [\epsilon, T]
$$

How can we guarantee $f(\cdot, \epsilon)$ is identity ? __Parameterization__

$$
f_\theta(x,t)=c_{\text{skip}}(t)x+c_{\text{out}}(t)F_\theta(x,t)
$$

where $F_\theta$ is a neural network, $c_{\text{skip}}(\epsilon)=1$ and $c_{\text{out}}(\epsilon)=0$.

Given such $f_\theta$, how to sample ? Theortically, one step forward is enough. We can also use multistep consistency sampling by using decreasing step size $\tau_n$

$$
\hat{x}_{\tau_n}\leftarrow x+\sqrt{\tau_n^2-\epsilon^2}z\quad z\sim \mathcal{N}(0,I)\\
x \leftarrow f_\theta(\hat{x}_{\tau_n}, \tau_n)
$$

### Distillation

Given pretrained score function $s_\phi(x,t)$, i.e. corresponding update function of ODE $\Phi(x,t;\phi)=\mu(x,t)-\frac{1}{2}\sigma^2(t)s_\phi(x,t)$, how can we distill $f_\theta$ ?

Discretizing the time horizon $[\epsilon, T]$ into $N-1$ intervals, with boundaries $t_1=\epsilon, t_N=T$, then

$$
\hat{x}_{t_n}^\phi=x_{t_{n+1}}+(t_n-t_{n+1})\Phi(x_{t_{n+1}}, t_{n+1},\phi)
$$

where $\Phi(\cdot,\cdot;\phi)$ is update function of ODE. To guarantee consistency, we need to minimize the following loss

$$
\mathcal{L}_{CD}(\theta,\theta^-;\phi)=\mathbb{E}_{n\sim U[1,N-1], x\sim p_{\text{data}},x_{t_{n+1}}\sim\mathcal{N}(x;t_{n+1}^2 I)}\left[\lambda (t_n)d(f_\theta(\hat{x}_{t_n}^\phi,t_n),f_{\theta^-}(x_{t_{n+1}},t_{n+1}))\right]
$$

where $\theta^-$ is running average of $\theta$.

### Isolation Training

Without pretrained score function, we can use unbiased estimator of score function

$$
\hat{s}_\phi(x,t)=\nabla \log p_t(x_t)=\mathbb{E}_{x\sim  p_{\text{data}},x_t\sim \mathcal{N}(x;t^2 I)}\left[\frac{x-x_t}{t^2}|x_t\right]
$$

using consistency loss

$$
\mathcal{L}_{CT}(\theta,\theta^-)=\mathbb{E}_{n\sim U[1,N-1], x\sim p_{\text{data}},z\sim\mathcal{N}(0;I)}\left[\lambda (t_n)d(f_\theta(x+t_{n+1} z,t_{n+1}),f_{\theta^-}(x+t_{n} z,t_{n}))\right]
$$

We can prove when $\sup_n (t_n-t_{n+1})\to 0$, with ground truth score function, $\mathcal{L}_{CT}$ converges to $\mathcal{L}_{CD}$.

<!-- ## Discussion -->

<!-- What is the difference between consistency models and DDIM ? -->

<!-- Consistency models actually learning a map from  -->