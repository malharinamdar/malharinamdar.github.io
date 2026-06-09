---
layout: page
title: DiabetesCare AI
description: diabetes risk prediction with personalised GenAI guidance
img: assets/img/diabetes.png
# years: [2020, 2019, 2018]
importance: 2
category: previous
---

DiabetesCare AI predicts a patient's risk of diabetes from their health metrics and then uses GenAI to turn that prediction into something genuinely useful — personalised dietary and lifestyle advice.

On the prediction side, the model is a **Random Forest classifier**, tuned with GridSearchCV and trained on a dataset of around 100,000 records. It reaches about **0.94 accuracy** on the test set. The user enters indicators like age, gender, hypertension, heart disease, smoking history, height, weight, HbA1c level and blood glucose, with BMI computed automatically, and the app returns a risk prediction along with a simple visualisation of the result.

Where it goes beyond a plain prediction is the **Gemini-1.5-Flash** integration. For patients flagged at risk, the model generates tailored lifestyle and dietary suggestions, including pointers to relevant resources and hospitals in India. There's also a **chatbot** built on the same model, so patients can ask follow-up questions and get advice in a conversational way, with their query history kept through the session.

The whole thing is built as a **Streamlit** web app, using scikit-learn for the model and Seaborn/Matplotlib for the visualisations.

website now live at <a href="https://diabetescare-ai-tech.streamlit.app/">DiabetesCare AI</a>

code available at <a href="https://github.com/malharinamdar/DiabetesCare-AI">repo</a>
