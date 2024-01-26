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
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/e6d23d09-9ab2-49d9-9d29-3895d5051ca1)


<span style='color:green;font-weight:bold'> Lack of instruction tuning data </span>
<br>
최근 NLP 에서는 LLM 의 강력함을 목격하였다. 
그 중심에는 두 가지 key component 가 있는데, (1)LLM 모델과 (2) human-written instruction data (e.g. PROMPTSOURCE, SUPERNATURALINSTURCTIONS, SUPERNI) 이다.
<span style='background-color: #ffdce0'> 그러나 instruction data 를 collecting 하는 것은 매우 costly 하고, annotator 가 입맛에 맞는 task 는 유명한 task들이기에 limited diversity 를 갖는다. </span>
따라서 instruction tuning process 를 위한 대체 방안이 필요하다.

<span style='color:green;font-weight:bold'> SELF-INSTRUCT </span>
<br>
이 논문에서는 모델 자체의 instructional signal 로 부터 instruction tuning 을 진행하는 semi-automated process 인 <span style='background-color: #dcffe4'> SELF-INSTRUCT </span> 방법을 제안한다. 
위의 그림이 SElF-INSTRUCT 방법론의 overall process 이다.
- (1) 우선 제한된 수의 seed task set (human-written) 으로 시작하여, 새로운 new task 를 위한 instruction 을 생성하게 한다. 이 과정에서 기존의 instruction 들의 collection 으로 부터, new task 를 위한 broad-coverage instruction 을 생성하게 한다.
- (2) 이후 생성된 instruction 을 바탕으로, 모델은 input 과 output 을 생성하게 된다.
- (3) 마지막으로, vlow-quality 와 repeated instruction 을 제거하기 위한 다양한 heuristic 을 통해 filtering 을 진행한다.
- (4) 이 과정은 task 의 수가 원하는 정도로 많아질 때 까지 반복된다.

<span style='color:green;font-weight:bold'> Experiments </span>
<br>
저자들은 SELF-INSTRUCT 방법을 Vanilla GPT-3 에 적용하였다.
이 방법을 통해 52K instruction 과 82K input-output pair 를 생성하였다.



<span style='color:green;font-weight:bold'> 초록색볼드체 </span>

<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
