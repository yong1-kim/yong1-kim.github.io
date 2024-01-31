---
layout: post
title:  "[EMNLP2023] EXPLORE-INSTRUCT: Enhancing Domain-Specific Instruction Coverage through Active Exploration"
date:   2024-01-31 17:47:00 +0900
use_math: true
categories: [LLM, PLM]
---

[[pdf]](https://aclanthology.org/2023.emnlp-main.587.pdf) &emsp;
[[github]](https://github.com/fanqiwan/Explore-Instruct)

**Fanqi Wan<sup>1*</sup>, Xinting Huang<sup>2†</sup>, Tao Yang<sup>1</sup>, Xiaojun Quan<sup>1†</sup>, Wei Bi<sup>2</sup>, Shuming Shi<sup>2</sup>**
<br><sup>1</sup> School of Computer Science and Engineering, Sun Yat-sen University, China <sup>2</sup> Tencent AI Lab &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/2aa06fa7-6a45-4214-8ce2-bc4db30177f7)

## Abstract
- (**Diversified Instruction Tuning**) Instruction Tuning 은 broader specturm 의 task 로 diversity 를 학습할 수 있다.
- (<span style='background-color: #ffdce0'> Lack of Data </span>) 그러나, 현존하는 data 로는 individual domain 에 대한 data 까지의 coverage 는 불가능하다.
- (<span style='color:green;font-weight:bold'> EXPLORE-INSTRUCT </span>) 이 논문에서는 LLM 의 exploration 을 통해, domain-specific instruction-tuning 를 위한 data coverage 를 발전시키는 방법론을 제안한다. 이 방법론은 몇몇 대표적인 domain 으로부터, diversified and domain-focused instruction-tuning data 를 생성할 수 있다.
- (**Experiment**) 생성한 data 에 대한 평가에서, 넓은 diversity 를 커버할 수 있는 것을 확인할 수 있으며, baseline 에 적용되었을 때, domain sepcific data enahncement 를 보였다.

## 1. Introduction




<span style='color:green;font-weight:bold'> 초록색볼드체 </span>

<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
