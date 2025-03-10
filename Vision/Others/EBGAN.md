# Energy-based Generative Adversarial Network (EBGAN)

## Theory

Use EBM as generator and discriminator in GAN

Given a margin $m$, data sample $x$ and generated sample $G(z)$, the loss function is defined as:
$$
L_D(x,z)=D(x)+max(0,m-D(G(z)))\\
L_G(z)=D(G(z))
$$

Define $V(G,D)=\int_{x,z}L_D(x,z)p(x)p(z)dxdz$ and $U(G,D)=\int_{z}L_G(z)p(z)dz$.
The Nash equilibrium of $D$ and $G$ satisfies:
$$
V(G^*,D^*)\leq V(G^*,D)\\
U(G^*,D^*)\leq U(G,D^*)
$$
for all $D$ and $G$. 
> THM: $p_{G^*}=p_{data}$ almost everywhere, $V(G^*,D^*)=m$
> THM: Nash equilibrium of $D$ and $G$ exists and characterized by $p_{G^*}=p_{data}$ almost everywhere, and $D^*(x)=\gamma\in [0,m]$ almost everywhere.

## Method

The discriminator $D$ is structured as an AE
$$
D(x)=\|x-Enc(Dec(x))\|_2
$$

Add a regularizer: pull-away term
$$
\frac{1}{N(N-1)}\sum_{i=1}^N\sum_{j\neq i}^N \left(\frac{v_i\cdot v_j}{\|v_i\|\|v_j\|}\right)^2
$$