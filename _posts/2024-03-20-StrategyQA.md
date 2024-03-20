---
layout: post
title:  "[TACL2021] Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies"
date:   2024-03-20 18:00:00 +0900
use_math: true
categories: [Retrieval, LLM, PLM]
---

[[pdf]](https://aclanthology.org/2021.tacl-1.21.pdf)  &emsp;
[[code and dataset]](https://allenai.org/data/strategyqa)

**Mor Geva<sup>1,2</sup>, Daniel Khashabi<sup>2</sup>, Elad Segal<sup>1</sup>, Tushar Khot<sup>2</sup>, Dan Roth<sup>3</sup>, Jonathan Berant<sup>1,2</sup>**
<br><sup>1</sup> Tel Aviv University <sup>2</sup> Allen Institute for AI <sup>3</sup> University of Pennsylvania &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/cb198cb6-b314-412b-9770-4585a63fb1fb)

# Abstract
- (**Explicit Multihop reasoning**) 현재 Multihop reasoning 의 큰 단한계점은 question 이 explicit 하다는 것이다.
- (<span style='color:green;font-weight:bold'> StrategyQA </span>) 이 논문에서는 question 속의 reasoning step 이 implicit 하게 내재되어 있는 StrategyQA 라는 QA benchmark 를 제시한다. 저자들은 term-based priming 기법을 통해 annotator 들로 하여금 창의적인 질문을 생성하게 하였고, adversarial filtering 과정을 거쳐 벤치마크를 생성하였다. 
- (**Statistics and Analysis**) 2,780 example 에 각각 decomposition 과 evidence paragrah 를 포함한다. StrategyQA 는 short, topic-diverse 하면서 넓은 범위의 strategy 를 cover 하고, 87% 점수의 human score 와 66% score 의 baseline score 를 report 한다.

## 1. Introduction

## 2. Strategy Qeustions
#2.1. Desiderata
#2.2. Decomposing Strategy Questions

## 3. Data Collection Pipeline
# 3.1. Creative Question Writing (CQW)
# 3.2. Strategy Question Decomposition (SQD)
# 3.3. Evidence Matching (EVM)
# 3.4. Data Verification Mechanisms

## 4. The STRATEGYQA Dataset
# 4.1. Dataset Statistics
# 4.2. Data Quality
# 4.3. Data Diversity
# 4.4. Human Performance


## 5. Experimental Evaluation
# 5.1. Baseline Models
# 5.2. Results


## Concclusion
```
We present STRATEGYQA, the first dataset of implicit multi-step questions requiring a widerange of reasoning skills. To build STRATEGYQA, we introduced a novel annotation pipeline for eliciting creative questions that use simple language, but cover a challenging range of diverse strategies. Questions in STRATEGYQA are annotated with decomposition into reasoning steps and evidence paragraphs, to guide the ongoing research towards addressing implicit multi-hop reasoning.
```

<span style='color:green;font-weight:bold'> 초록색볼드체 </span>
<br>
<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
