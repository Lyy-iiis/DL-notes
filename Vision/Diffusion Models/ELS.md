# An analytic theory of creativity in convolutional diffusion models

*The ideal score machine only memorizes*

However, our diffusion model $s_t(\phi)=M_t[\phi]$ is not ideal, and we use CNN as backbone
It mainly follows two inductive biases:
1. **Locality**: for all images $\phi$ and pixel locations $x$, $M_t[\phi](x)=M_t[\phi_{\Omega_x}](x)$
2. **Equivariance**: for any $U\in G$, $M_t[U\phi]=U\cdot M_t[\phi]$

Based on these two inductive biases, we can find ideal score with optimal MMSE approximation

Experiments: fantasy