# Introduction to Flow Matching

## Residual Flow

Sample: $x\sim q_0,y=\phi(x)$, the density

$$
p_1(y) = \frac{q_0(x)}{|\det(\frac{\partial \phi(x)}{\partial x})|} 
$$

A general $\phi$ is hard to compute Jacobian, so we find better structure for $\phi$

Full-rank residual: $\phi_k(x) = x + \delta u_k(x)$

$$
\phi = \phi_K\circ \cdots \circ \phi_1\\
\log q(y) = \log p(\phi^{-1}(y)) + \sum_{k=1}^K \log |\det(\frac{\partial \phi_k^{-1}}{\partial x_{k+1}}(x_{k+1}))|
$$

Continous time: when $\delta\to 0$

$$
\frac{d x_t}{dt} = \frac{\phi(x)-x}{\delta} = u(x) 
$$

define

$$
x_t = \phi_t(x_0) = x_0 + \int_0^t u(x_s) ds
$$

then 

$$
\frac{d \phi_t}{dt} = u_t(\phi_t(x_0)) 
$$

How to compute the Jacobian ? 流守恒方程. 

$$
\frac{\partial}{\partial t} p_t(x_t) = - (\nabla\cdot (u_t p_t)) (x_t)
$$

Using $\frac{d}{dt} = \frac{\partial}{\partial t} + u_t\cdot \nabla$, we have
$$
\frac{d}{dt} \log p_t(x_t) = -(\nabla\cdot u_t) (x_t)
$$

i.e.

$$
\frac{d}{dt} \begin{bmatrix}
x_t\\
\log p_t(x_t)
\end{bmatrix} = \begin{bmatrix}
u_\theta(x_t,t)\\
-div(u_\theta(x_t,t))
\end{bmatrix}
$$