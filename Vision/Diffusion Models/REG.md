# REG: Rectified Gradient Guidance for Conditional Diffusion Models

## Motivation

Original motivation for guidance is that we want to sample from

$$
\overline{p}_\theta(x_0|y)\propto p_\theta(x_0|y) \cdot R_0(x_0,y)
$$

where $R_0(x_0,y)$ is reward, different for different guidance methods
$$
CG: R_0(x_0,y) = [p_\phi(y|x_0)]^w\\
CFG: R_0(x_0,y) = \left[\frac{p_\theta(x_0|y)}{p_\theta(x_0)}\right]^w\\
AutoG: R_0(x_0,y) = \left[\frac{p_\theta(x_0|y)}{p_{\theta_{\text{old}}}(x_0|y)}\right]^w
$$

But practical implementation is

$$
\overline{p}_\theta(x_t|y)\propto p_\theta(x_t|y) \cdot R_t(x_t,y)
$$

where

$$
CG: R_t(x_t,y) = [p_{\phi}(y|x_t)]^w\\
CFG: R_t(x_t,y) = \left[\frac{p_\theta(x_t|y)}{p_\theta(x_t)}\right]^w\\
AutoG: R_t(x_t,y) = \left[\frac{p_\theta(x_t|y)}{p_{\theta_{\text{old}}}(x_t|y)}\right]^w
$$

the optimal reward function

$$
\overline{\epsilon}_{\theta,t} = \epsilon_{\theta,t} - \sqrt{1-\overline{\alpha}_t} \nabla_{x_t} \log R_t(x_t,y)
$$

**BUT given $R_t$, $R_{t-1}$ is implicitly defined, because $\overline{\epsilon}_{\theta,t}$ defines relation between $\overline{p}_\theta(x_t|y)$ and $\overline{p}_\theta(x_{t-1}|y)$**

$$
R_{t-1}(x_{t-1},y) \propto \frac{\mathbb{E}[\mathcal{N}(x_{t-1}|\overline{\mu}_{\theta,t},\sigma_t^2 I)R_t(x_t,y)]}{\mathbb{E}[\mathcal{N}(x_{t-1}|{\mu}_{\theta,t},\sigma_t^2 I)]}\\
\overline{\mu}_{\theta,t} = \mu_{\theta,t} + \frac{1-\alpha_t}{\sqrt{\alpha_t}}\nabla_{x_t} R_t(x_t,y)
$$

Two disadvantages:
1. The $\overline{p}_\theta(x_t|y)\propto p_\theta(x_t|y) \cdot R_t(x_t,y)$ is invalid as it imposes excessive constraints that
may not be satisfied (Why ?)
2. $\overline{p}_\theta(x_0|y)\propto p_\theta(x_0|y) \cdot R_0(x_0,y)$ is also invalid, because $R_1,...,R_T$ is identity implies $R_0$ is identity
3. If $R_T$ is identity, then all $R_t$ are identity. But since $x_T$ is Gaussian, $R_T$ may be identity (I think this make sense)

## Method

Joint distribution

$$
\overline{p}_\theta(x_{0:T}|y) \propto p_\theta(x_{0:T}|y) \cdot R_0(x_0,y)
$$

define induced expected reward $E_t(x_t,y)$ as

$$
E_t(x_t,y) = \int p_\theta(x_0|x_t,y)R_0(x_0,y)dx_0
$$

> THM: Transition kernel $\overline{p}_\theta(x_t|x_{t+1},y)$ is determined
> $$
> \overline{p}_\theta(x_t|x_{t+1},y) = \frac{E_t(x_t,y)}{E_{t+1}(x_{t+1},y)} p_\theta(x_t|x_{t+1},y)
> $$
> and
> $$
> \overline{p}_\theta(x_{t}|y) = \frac{E_t(x_t,y)}{E(y)}p_\theta(x_t|y)
> $$
> The noise prediction network is
> $$
> \overline{\epsilon}_{\theta,t}^* = \epsilon_{\theta,t} - \sqrt{1-\overline{\alpha}_t} \nabla_{x_t} \log E_t(x_t,y)
> $$

Tricks:
How to sample from $\overline{p}_\theta(x_T|y)$ ? $E_T(x_T,y)$ is intractable, but we can approximate $E_T(x_T,y)$ by $E_T(y)$, i.e. sample from $\mathcal{N}(0,I)$

$\overline{\epsilon}_{\theta,t} = \epsilon_{\theta,t} - \sqrt{1-\overline{\alpha}_t} \nabla_{x_t} \log R_t(x_t,y)$ is an approximation of $\overline{\epsilon}_{\theta,t}^*$, **but we can explore alternative approximation**. 

### REG

$$
\overline{\epsilon}_{\theta,t}^{\text{REG}} = \epsilon_{\theta,t} - \sqrt{1-\overline{\alpha}_t} \nabla_{x_t} \log R_t(x_t,y) \odot \left(1-\sqrt{1-\overline{\alpha}_t} \frac{\partial (1^T\cdot \epsilon_{\theta,t})}{\partial x_t}\right)
$$