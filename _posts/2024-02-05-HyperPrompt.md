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

# Abstract
- (<span style='color:green;font-weight:bold'> Hyperprompt </span>) 이 논문에서는 Transformers 속의 self-attention 에 prompt-based task-conditioning architecture 인 **Hyperprompt** 를 제안한다.
- (**Global memory**) HyperNetwork 를 활용하는 hyperprompt 는 task 간의 정보 교환을 해주는 역할에 더불어, task 의 global memory 의 역할을 한다는 것을 보인다.
- (**Efficiency**) 단지 0.14% 의 추가적인 param 만으로 T5 등의 multi-task learning baseline 과 비교하여 competetive 한 성능을 보인다.

<span style='color:green;font-weight:bold'> 초록색볼드체 </span>

<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
