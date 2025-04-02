# Diffusion-LM Improves Controllable Text Generation

## Controlable Generation for Text

Given control variable $c$, do controlable generation using
$$
p(w|c) = p_{lm}(w)p(c|w)
$$

## Diff-LM

How to apply continuous diffusion model to discrete text ?

Forward discrete words $w$ to $x_0$ by adding noise, round $x_0$ to $w$ using a softmax distribution $p_\theta$ in reverse process

$$
q(x_0|w)=\mathcal{N}(EMB(w),\sigma_0I)\\
p_\theta(w|x_0)=\prod_{i=1}^n p_\theta(w_i|x_{0,i})
$$

add corresponding objective

$$
\mathcal{L}_{vlb}^{e2e}(w)= \mathbb{E}_{q_\phi(x_0|w)}\left[\mathcal{L}_{vlb}(x_0)+\log q_\phi(x_0|w)-\log p_\theta(w|x_0)\right]\\
\mathcal{L}_{simple}^{e2e}(w) = \mathbb{E}_{q_\phi(x_{0:T}|w)}\left[\mathcal{L}_{simple}(x_0)+\|EMB(w)-\mu_\theta(x_1,1)\|^2-\log p_\theta(w|x_0)\right]
$$

For ideal denoising networks, $x_0$ should be some $EMB(w)$, but this loss only force predicted $x_0$ to be close to $EMB(w)$ in $t\to 0$.

Instead of predict $x_{t-1}$, we can predict $x_0$ directly
$$
\mathcal{L}_{simple}^{e2e}(x_0)=\sum_{t=1}^T\mathbb{E}_{x_t}\left[\|f_\theta(x_t,t)-x_0\|^2\right]
$$
Inference
$$
x_{t-1}=\sqrt{\overline{\alpha_t}}\text{Clamp}(f_\theta(x_t,t))+\sqrt{1-\overline{\alpha_t}}\epsilon
$$
where Clamp implies find nearest neighbor in $EMB(w)$

### Controlable Generation

$$
\nabla_{x_{t-1}}\log p(x_{t-1}|x_t,c) = \nabla_{x_{t-1}}\log p(x_{t-1}|x_t) + \nabla_{x_{t-1}}\log p(c|x_{t-1})
$$

we can add parameter $\lambda$ to trades off fluency and control

$$
\lambda\log p(x_{t-1}|x_t) + \log p(c|x_{t-1})
$$

### Decoding

Minimum Bayes Risk (MBR) decoding
$$
\hat{w} = \arg\min_{w\in S}\sum_{w'\in S}\frac{1}{|S|}\mathcal{L}(w,w')
$$
