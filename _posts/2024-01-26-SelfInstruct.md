---
layout: post
title:  "[ACL2023] SELF-INSTRUCT: Aligning Lnaugage Models with Self-Generated Insructions"
date:   2024-01-26 17:00:00 +0900
use_math: true
categories: [LLM, PLM]
---

[[pdf]](https://arxiv.org/pdf/2212.10560.pdf) &emsp;
[[github]](https://github.com/yizhongw/self-instruct)

**Yizhong Wang <sup>♣</sup>, Yeganeh Kordi <sup>♢</sup>, Swaroop Mishra <sup>♡</sup>, Alisa Liu <sup>♣</sup> Noah A. Smith <sup>♣+</sup>, Daniel Khashabi <sup>♠</sup>, Hannaneh Hajishirzi <sup>♣+</sup>**
<br><sup>♣</sup> University of Washington <sup>♢</sup> Tehran Polytechnic <sup>♡</sup> Arizona State University <sup>♠</sup> Johns Hopkins University <sup>+</sup> Allen Institute for AI &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/16fe65d2-8530-4578-a620-797d64dcf87a)

## Abstract
- (**Instruction Tuning**) Instruction 에 respond 할 수 있게 language model 을 finetuning 하는 instruction tuning 을 통해 새로운 task 에 대한 높은 일반화 성능을 부여할 수 있다.
- (**Lack of dataset**) 그러나, human-written instruction data 는 그 양과 다양성, 창의성(creativity) 가 부족하다.
- (<span style='color:green;font-weight:bold'> SELF-INSTRUCT </span>) 저자들은 LLM 이, 자신이 생성한 generation 을 bootstrapping 하는 기법을 통해 instruction-following 능력을 개선시키는 framework 을 제시한다.
- (**Pipeline**) Pipiline 은 instruction, input, output sample 을 generate 한 뒤, invalid 하거나 이전과 유사한 것들을 filter 하는 형식이다.
- (**Experiment**) 이 방법으로 vanilla GPT-3 에 적용했을 때, private user data 와 human annotation 을 배운 InstructGPT (text-davinci-001) 를 뛰어넘는 성능을 보인다.
- (**Broader Impact**) SELF-INSTRUCT 는 <span style='background-color: #dcffe4'> almost annotation-free method </span> 로 PLM 은 instruction 에 aligning 할 수 있게 하며, 추후 instruction tuning 에 사용될 수 있는 synthetic dataset 을 생성하여 release 하였다.

## Introduction






<span style='color:green;font-weight:bold'> 초록색볼드체 </span>

<span style='background-color: #dcffe4'> 초록색배경 </span>
