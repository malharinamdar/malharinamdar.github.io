---
layout: page
title: Vaidya Nidaan
description: MRI diagnosis for alzheimer's with biomarkers and explainability using ml and medical imaging
img: assets/img/vaidya-nidaan.png
# years: [2020, 2019, 2018]
importance: 2
category: previous
---

Vaidya Nidaan is an end-to-end Alzheimer's diagnostic platform I built for the PICT Techfiesta hackathon. The goal was to make something that could genuinely assist a doctor — not just predict a label, but back that prediction up with a visual explanation, quantitative measurements, and a report.

It pulls together a few different pieces:

- **MRI classification** — a VGG-19 convolutional network classifies the MRI scan, reaching around 95% accuracy on Alzheimer's detection.
- **Explainability with Grad-CAM++** — instead of just giving a prediction, the system generates heatmaps that highlight which regions of the brain the model was actually focusing on. Watching it consistently light up clinically relevant areas like the hippocampus is what first got me genuinely interested in interpretability.
- **Biomarker extraction with FSL** — using the FMRIB Software Library, the pipeline automatically computes quantitative measurements from the scan, such as hippocampal volume and ventricular size, which are the kinds of biomarkers a clinician actually looks at.
- **Multilingual chatbot** — a GPT-4-turbo powered assistant with retrieval, able to answer questions in English, Hindi, Marathi and other regional Indian languages.
- **Automated reports** — the system pulls everything together into a structured medical report with the biomarker values and findings.

All of this is wrapped in a full web application — a React frontend for uploading scans and viewing results, and a Node.js backend handling the model, storage and authentication.

Building it meant moving between deep learning, medical imaging, explainability and full-stack development, which made it one of the most complete projects I've worked on.

Edit: (23/02/25) : Stood 3rd among 400+ teams at Techfiesta Hackathon 2025 🥳

code available at <a href="https://github.com/malharinamdar/vaidya-nidaan">repo</a>
