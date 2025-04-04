# Distilling Diffusion Models into Conditional GANs

## Motivation

1. Find correspondence between noises and images by simulating ODE
2. Do noise-to-image pair task using conditional GAN

## Method

### Distillation

Trivial Distillation:
$$
\mathcal{L}_{\text{distill}}^{\text{ODE}} = \mathbb{E}_{\{z,c,x\}}[d(G(z,c),x)]
$$

Two way to improve:
1. Scale up ODE pair dataset
2. Use perceptual loss instead of pixel loss

By using perceptual loss, we can use LPIPS loss
$$
\mathcal{d}_{\text{LPIPS}}(x,y) = \mathcal{L}(F(Decode(x)),F(Decode(y)))
$$
where Decode maps latent space to image space(8x upsampling)

In order to improve the efficiency, we can compute loss in latent space
$$
\mathcal{d}_{\text{LatentLPIPS}}(x,y) = \mathcal{L}(F(x),F(y))
$$
But single image reconstruction doesn't converge under LatentLPIPS, but we can use Ensembled-LatentLPIPS
$$
\mathcal{d}_{\text{E-LatentLPIPS}}(x,y) = \mathbb{E}_{\mathcal{T}}[\mathcal{L}(F(\mathcal{T}(x)),F(\mathcal{T}(y)))]
$$

### Conditional Diffusion Discriminator

Condition on noise $z$ and other condition $c$, train discriminator using 
$$
\min_G \max_D \mathbb{E}_{\{z,c,x\}}[\log D(z,c,x)] + \mathbb{E}_{\{z,c\}}[\log(1-D(z,c,G(z,c)))]
$$
and generator using
$$
\min_G \mathcal{L}_{GAN}(G,z,c,x)=\min_G -\mathbb{E}_{\{z,c\}}[\log D(z,c,G(z,c))]
$$

the final loss is
$$
\mathcal{L}_G = \mathcal{L}_{\text{E-LatentLPIPS}} + \lambda_{GAN} \mathcal{L}_{GAN}
$$

Tricks:
1. Init discriminator from pretrained diffusion model 
2. Apply R1 regularization only to one single sample in the batch
3. Multi-scale in-and-out discriminator
4. Data augmentation: random replace a portion of generated latents with random, unrelated latents