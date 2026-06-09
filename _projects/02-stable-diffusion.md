---
layout: page
title: Stable diffusion from scratch
description: rebuilding the architecture behind modern image generation, component by component
img: assets/img/stable.png
# years: [2020, 2019, 2018]
importance: 2
category: current
---

I built this project because I wanted to actually understand how Stable Diffusion works, not just use it through a library. So instead of importing a ready-made pipeline, I reimplemented the inference architecture of Stable Diffusion v1.5 from scratch in PyTorch, piece by piece, following the original papers.

The model is made of several parts that each do a specific job, and building them separately is what made the whole thing click for me:

- **Variational Autoencoder (VAE)** — diffusion doesn't happen on the raw image, it happens in a compressed *latent space*. The VAE encoder shrinks a 512×512 image down into a small latent representation, and the decoder turns it back into pixels at the end.
- **CLIP text encoder** — this turns the text prompt into embeddings that condition the generation, so the image actually reflects what you asked for.
- **U-Net** — the core of the system. It's the network that, step by step, looks at a noisy latent and predicts the noise to remove. I implemented the attention blocks here too, including the cross-attention that lets the text prompt influence the image.
- **DDPM sampler** — the scheduler that runs the denoising loop, starting from pure noise and gradually turning it into a coherent latent over many steps.

The pipeline ties these together for both text-to-image and image-to-image generation, with classifier-free guidance to control how strongly the prompt is followed. Working through each component — how the latent space connects to the U-Net, how the noise schedule works, how the text conditioning actually reaches the image — gave me a much deeper feel for diffusion models than any amount of reading would have.

I plan to add the notes I took while building this, and hopefully a short blog post walking through the architecture.

code available at <a href="https://github.com/malharinamdar/lumos">repo</a>
