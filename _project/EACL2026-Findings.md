---
layout: project_single
title:  "Findings of EACL 2026"
slug: "EACL2026-Findings"
---
**Don't Judge Code by Its Cover: Exploring Biases in LLM Judges for Code Evaluation**

Jiwon Moon\*, Yerin Hwang\*, Dongryeol Lee, Taegwan Kang, **Yongil Kim**, Kyomin Jung

**Abstract:**
With the growing use of large language models (LLMs) as evaluators, their application has expanded to code evaluation tasks, where they assess the correctness of generated code without relying on reference implementations. This research investigates whether LLM judges can fairly evaluate semantically equivalent code that differs only in superficial ways like variable names, comments, or formatting. We identified six categories of potential bias in code evaluation and tested multiple LLMs across five programming languages. Our findings demonstrate that all tested LLM judges exhibit susceptibility to both positive and negative biases, leading to either inflated or unfairly low scores. Notably, even when prompted to generate test cases before scoring, the LLM judges remained vulnerable to these biases, underscoring the necessity for developing more dependable code evaluation methodologies.

[\[Paper\]](https://arxiv.org/abs/2505.16222)
