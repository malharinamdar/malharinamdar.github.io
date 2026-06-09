---
layout: page
title: Circuit-Lens
description: peeking inside small language models with mechanistic interpretability
img: assets/img/circuit-lens.jpg
importance: 1
category: current
---

After we finished the [Regional-TinyStories](https://aclanthology.org/2025.findings-ijcnlp.142/) work, I wanted to go a level deeper and actually dive into the internals of the SLM checkpoints we had trained. We had built small language models for Hindi, Marathi and Bangla and measured how well they generated text, and I was really curious to explore what was happening *inside* them — how they were representing language across their layers. Circuit-Lens started as my attempt to open them up and find out.

It is a hook-based mechanistic interpretability framework built around the SLMs from the paper. The idea is simple: instead of only looking at what comes out of the model, I attach forward hooks that record what happens at every layer as a sentence passes through, and then run a set of analyses on top of those recordings.

{% include figure.html path="assets/img/circuit-lens.jpg" class="img-fluid rounded z-depth-1" zoomable=true caption="The residual stream: every block reads from a running vector and adds its result back, which is what makes layer-by-layer inspection possible." %}

## The idea behind it

The whole thing rests on one fact about transformers, shown above. Every layer doesn't overwrite the model's internal state, it *adds* to a shared running vector called the residual stream. Because of that, you can stop at any layer and ask "what does the model believe right now?" — the maths still lines up. Circuit-Lens wraps a nanoGPT-style model and exposes that stream, the attention patterns, and the MLP activations at every layer, without changing the underlying model at all. One small but important detail: I had to switch off the fused flash-attention kernel, because it never actually materialises the attention matrix, and that matrix is exactly what you need to see which tokens are attending to which.

## What it can do

Once the internals are exposed, the framework implements a handful of standard interpretability techniques:

- **Logit lens** — apply the model's output projection to the residual stream at *each* layer, so you can watch the next-token prediction form and sharpen as it moves through the network.
- **Activation patching** — the causal workhorse. Run a clean sentence and a corrupted one, then copy a single activation from the clean run into the corrupted run. If the output flips, that location is *causally* responsible for the behaviour, not just correlated with it.
- **Induction-head detection** — find the attention heads that implement the `[A][B] … [A] → [B]` pattern, the heads widely believed to underlie in-context learning.
- **Gender–verb agreement circuits** — the part I find most interesting. Hindi and Marathi verbs change with the subject's gender (लड़का जाता है vs लड़की जाती है). I use activation patching across layers to locate where in the model that gender signal actually lives.
- **Neuron analysis** — search for individual MLP neurons that fire for specific features, like masculine vs feminine forms.
- **Activation steering** — compute a direction in activation space (say, formal minus informal) and add it back during generation to push the model's outputs one way or the other.
- **Cross-lingual comparison** — run the same analyses across the Hindi, Marathi and Bangla models and ask whether they develop similar internal structure.

## Where it stands

This is honest, early-stage research code. The machinery works and produces structured results, but the findings are still a work in progress — most of the analyses run on small hand-built prompt sets, and the next step is scaling them up properly. Building it gave me a lot of hands-on time with transformer internals, and it's the project that pushed me toward interpretability as the direction I most want to keep working in.

code available at <a href="https://github.com/malharinamdar/circuit-lens">repo</a>
