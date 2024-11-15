---
layout: post
title: "Overleaf = Dissertation + Papers"
author: Sebastian Daza
comments: true
date: 2019-11-01
---


It's been a while since my last post. It's time to catch up. I am working on my dissertation and using a handy workflow where you can both writing independent papers and chapters of your thesis using Overleaf and Github. The advantage of this workflow is:

- Keep your in you dissertation or book repository all the files necessary to run analysis
- Keep independent papers for each chapter, almost ready for reviewing and publishing
- Allow collaboration and reviews for each chapter via Overleaf
The set up a simple:

- I use [Overleaf](/www.overleaf.com) to write my papers and dissertation. Each chapter or article is an independent project.
- The structure of files would be something like:

```
project
│   dissertation.tex
│   ref.bib
│
└───ch01
│   │   manuscript.tex
│   │   ch01.bib
│   │
│   └───tables
│       │   table01.tex
│       │   table02.tex
│       │   ...
│
└───ch02
|   │   manuscript.tex
|   │   ch02.bib
|   |
│   └───tables
│       │   table01.tex
│       │   table02.tex
│       │   ...
```

- `dissertation.tex` is the main manuscript. In order
