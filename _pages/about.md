---
layout: about
title: about
permalink: /about/
description: research, machine learning and artificial intelligence
profile:
  align: right
  image: malhar_prof_pic.jpeg
  image_circular: false
  address: >
    <p style="font-style: italic; font-size: 0.9rem; margin-top: 8px; line-height: 1.4; color: #6c757d !important;" class="dark:text-gray-400">
      ⛰️ Among the hills of Mussoorie, Uttarakhand
    </p>
news: true
selected_papers: false # Set this to true to show your publication
social: false # Set this to true to show social icons
---

Hey, I'm Malhar, a third-year undergrad at PICT, Pune, working on AI systems that can reason through complex problems and serve humanity in meaningful ways.

My research interests center on healthcare AI, reasoning models, and multilingual language intelligence. I'm particularly drawn to building systems that improve accessibility in low-resource settings - whether that means developing diagnostic tools for underserved communities or creating language models, frameworks for mutlilingual settings.

### What I've worked on


**Vaidya Nidaan** — Led the development of an *Alzheimer's* diagnostic platform that integrates CNN-based analysis with medical imaging using FSL biomarker identification (hippocampal volume, white/gray matter ratios). The system employs *GradCAM* for visual interpretability and includes a multilingual RAG pipeline that generates structured medical reports grounded in research literature. This work secured third place among 400+ teams at PICT Techfiesta 2025.

**Medical Reasoning with Phi-3** — Fine-tuned Phi-3-mini-4k-instruct on the *MedQA-USMLE* dataset to teach structured clinical reasoning. Using QLoRA, I trained the model to generate explainable diagnostic chains-of-thought while engaging only 0.44% of its parameters. To explore preference-based alignment further, I implemented a GRPO trainer from scratch in PyTorch, working through the practical challenges of low-precision optimization.

[**Regional TinyStories**](https://arxiv.org/abs/2504.07989) — At [Vizuara AI Labs](https://vizuara.ai), I conducted research on multilingual language modeling under Dr. Raj Dandekar, extended Microsoft's TinyStories (2023) work for Indian regional languages. I trained Small Language Models (2M–150M parameters) from scratch for Hindi, Marathi, and Bengali. We developed a novel framework for the development and analysis of SLMs, tokenizer performance, linguistic complexity, machine translation performance and demonstrated that a 53M parameter model could achieve GPT-3.5-comparable results on short-story generation - a promising direction for accessible, high-quality models in Indian languages. This work is currently under review at ACL ARR 2025.

At [Froncort.AI](https://froncort.ai), Implemented a RLHF pipeline that converted expert reviewer feedback into heuristic reward signals, improving LLM output quality by 35% while working within tight compute constraints compliant with international regulatory and ontological standards. 

### When I'm Not Training Models..

I'm probably reading papers, books at odd hours or wandering through the hills. I'm drawn to the philosophical questions that emerge from AI's capabilities, but I'm most energized by the opportunity to build systems that genuinely help people. Music often accompanies this process — it’s become a quiet companion that helps me think through problems and find clarity.

If you're working on something in healthcare AI, language modeling or just ml in general, feel free to reach out! I'd be glad to hear from you.
