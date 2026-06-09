---
layout: page
title: mcq generator
description: turning documents into multiple-choice questions
img: assets/img/mcq.png
# years: [2020, 2019, 2018]
importance: 2
category: previous
---

This is a web app that takes a document and automatically generates multiple-choice questions from it. You upload a `.txt` or `.pdf`, pick how many questions you want and at what difficulty (easy, medium or hard), and it produces a quiz from the content.

My original plan was actually more ambitious on the NLP side — I wanted to build the question generation myself using a **T5** encoder-decoder model together with a **BERT** encoder to analyse the text and synthesise questions. I started down that path, but between the complexity involved and a few other constraints, I paused that approach for the time being.

For the working version, I took a more direct route and used the **Gemini-1.5-Flash** model to do the generation, wrapped in a simple **Streamlit** app. It's intentionally lightweight at this stage. I still plan to come back and build out the original transformer-based pipeline, and to add more features to the webapp.

website now live at <a href="https://mcq-generator-web.streamlit.app/">link</a>

Code available at <a href="https://github.com/malharinamdar/mcq-generator">repo</a>
