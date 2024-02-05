---
layout: post
title:  "[ICML2022] HyperPrompt: Prompt-based Task-Conditioning of Transformers"
date:   2024-02-05 09:08:00 +0900
use_math: true
categories: [LLM, PLM]
---

[[pdf]](https://proceedings.mlr.press/v162/he22f/he22f.pdf)


**Yun He <sup>1*</sup>, Huaixiu Steven Zheng <sup>1*</sup>, Yi Tay <sup>2</sup>, Jai Gupta<sup>2</sup>,  Yu Du <sup>2</sup>, Vamsi Aribandi <sup>2</sup>, Zhe Zhao <sup>2</sup>, YaGuang Li<sup>2</sup>, Zhao Chen<sup>3</sup>, Donald Metzler<sup>2</sup>, Heng-Tze Cheng<sup>2</sup>, Ed H. Chi<sup>2</sup>**
<br><sup>1</sup> Texas A&M University <sup>2</sup> Google Research <sup>3</sup> Waymo LLC. &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/096235d2-fe66-4059-b1c8-91e6726c03e1)

## Abstract
- (<span style='color:green;font-weight:bold'> Hyperprompt </span>) 이 논문에서는 Transformers 속의 self-attention 에 prompt-based task-conditioning architecture 인 **Hyperprompt** 를 제안한다.
- (**Global memory**) HyperNetwork 를 활용하는 hyperprompt 는 task 간의 정보 교환을 해주는 역할에 더불어, task 의 global memory 의 역할을 한다는 것을 보인다.
- (**Efficiency**) 단지 0.14% 의 추가적인 param 만으로 T5 등의 multi-task learning baseline 과 비교하여 competetive 한 성능을 보인다.

## 1. Introduction
<span style='color:green;font-weight:bold'> ▶HyperPrompt </span>
<br>
Soft Learnable memory token 으로 LLM 을 condition 하는 prompt tuning 이 주목을 받고 있다.
Pretrained model 은 frozen 한 채 빠르고 가볍게 학습할 수 있다는 장점을 갖고 있다.

이 논문에서는 Multi-task learning 을 위한 새로운 Prompt-tuning 방법론인 HyperPrompt 를 제안한다.
<span style='background-color: #dcffe4'> HyperPrompt 는 task-conditioned hyper-prompt 를 도입하여, prompt 를 통해 task-specific information 을 모델이 condition 할 수 있게 한다. </span>

<span style='color:green;font-weight:bold'> ▶HyperNetwork </span>
<br>
저자들은 이 hyperprompt 를 위하여, HyperNetwork 를 도입한다.
이 HyperNetwork 가 task-aware and layer-aware prompt 를 generation 한다.
보통 기존의 multi-task learning 방법들은 task 수에 linear 하게 param 이 증가하기 마련인데, <span style='background-color: #dcffe4'> HyperNetwork 를 활용하면 아주 적은 양의 추가적인 param 만으로 기존의 방법들과 competitive 한 성능을 보일 수 있어 효율적이다. </span>
그리고 이들은 **prompt generator 의 개념은 HyperNetwork 가 처음**이라고 주장한다.

이들은 기존의 prompt 학습 방식이나 adapter 와 같은 개념과 다르게 network 전체를 학습시키는 것이 중요하다고 한다. 그 이유로는 (1) 기존의 prompt-tuning 은 11B 이상의 LLM 에 대해서만 잘 적용이 되며, (2) adaptive param 만 학습한다고 해서 inference 에서 딱히 이득이 없다고 한다. 따라서, Network 를 전체 학습하여 성능을 높이는 것이 더 낫다고 판단한다.

<span style='color:green;font-weight:bold'> 초록색볼드체 </span>

<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
