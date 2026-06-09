---
layout: page
title: Phi-3 medical fine-tuning
description: adapting a small language model for medical question answering
img: assets/img/phi3-finetune.jpg
importance: 1
category: previous
---

This project was my way of getting properly hands-on with the modern fine-tuning stack. I took Microsoft's Phi-3-mini and adapted it to answer USMLE-style medical multiple-choice questions, going through the full pipeline rather than just calling an API.

{% include figure.html path="assets/img/phi3-finetune.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="The basic shape of fine-tuning: take a general model, train it on a curated dataset, and end up with a model specialised for the task." %}

## How it works

The base model is loaded in **4-bit precision** using bitsandbytes, which is what makes it possible to train a model this size on a single GPU. On top of that I use **QLoRA** — instead of updating all the model's weights, I train small low-rank adapter matrices that get injected into the attention and MLP layers. This keeps the number of trainable parameters tiny while still letting the model specialise. The whole thing runs through Hugging Face `transformers`, `peft` for the adapters, and `trl` for the training loops.

The training happens in two stages:

- **Supervised fine-tuning (SFT)** — the model first learns the format and reasoning style of medical QA from prompt/answer pairs.
- **Preference optimisation** — a second stage using a dataset of `chosen` vs `rejected` answer pairs, nudging the model toward the better-reasoned response. This is the same family of techniques used to align larger models, and it was the part I most wanted to understand by actually building it.

## What I took away from it

Getting QLoRA, 4-bit quantisation and preference training to all work together on Phi-3 took a fair bit of debugging, but that was exactly the point — I came out of it genuinely comfortable with how parameter-efficient fine-tuning and alignment-style training fit together, rather than just knowing the terms. The dataset of preference pairs and the full notebook are in the repo.

code available at <a href="https://github.com/malharinamdar/phi3-finetune">repo</a>
