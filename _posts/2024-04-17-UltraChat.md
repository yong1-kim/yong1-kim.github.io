---
layout: post
title: "[EMNLP2023] Enhancing Chat Language Models by Scaling High-quality Instructional Conversations"
date: 2024-04-17 13:20:00 +0900
use_math: true
categories: [LLM, PLM]
---

[[pdf]](https://aclanthology.org/2023.emnlp-main.183.pdf) &emsp;
[[github]](https://github.com/thunlp/UltraChat)

**Ning Ding<sup>1∗</sup>, Yulin Chen<sup>2,3∗</sup>, Bokai Xu<sup>4</sup>, Yujia Qin<sup>2,3</sup>, Shengding Hu<sup>2,3</sup>, Zhiyuan Liu<sup>2,3†</sup>, Maosong Sun<sup>2,3†</sup>, Bowen Zhou<sup>1†</sup>**
<br> <sup>1</sup> Department of Electronic Engineering, Tsinghua University, <sup>2</sup> Department of Computer Science and Technology, Tsinghua University, <sup>3</sup> BNRIST, IAI, Tsinghua University, <sup>4</sup> The Chinese University of Hong Kong, Shenzhen &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/02a7fabe-3a4c-4303-8fdb-17b6ee11d6a3)

## Abstract
- (**Finetuning chat Language model**) ChatGPT 와 같은 chat language model 을 instruction data 를 통한 fine-tuning 하는 것은, diversity 와 quality 가 받춰줄 때 좋은 성능을 끌어올릴 수 있는 방법이다.
- (<span style='color:green;font-weight:bold'> UltraChat </span>) Human query 를 포함하지 않은 large-scale instructional conversation 을 담고 있는 UltraChat 을 제안한다. 이 데이터셋은 scale, average length, diversity, coherence 등에서 우수성을 보인다.
- (**UltraLM**) UltraChat 을 활용하여 LLaMA 를 finetuning 하여, UltraLM 을 만들었고, WizardLM 과 Vicuna 를 포함한 open-source model 보다 좋은 성능을 보인다.

## 1. Introduction

<span style='color:green;font-weight:bold'> ▶ Chat LLM </span>
<br>
Large Language Model (LLM) 은 놀라울 만한 성능을 보이며, conversation 에 특화된 ChatGPT 와 같은 Chat LLM 은 선풍적인 인기를 끌고 있다.
현재 많은 open-source Chat LM 모델들이 공개되었지만, <span style='background-color: #ffdce0'> ChatGPT 나 GPT-4 는 고사하고 Vicuna 를 이기는 모델 조차 없다. (2023년 5월 기준) </span>

<span style='color:green;font-weight:bold'> ▶ UltraChat </span>
<br>
이 연구에서는 가장 단순한 방법으로 성능을 끌어올릴 수 있다고 믿는다: <span style='background-color: #dcffe4'> Quality and diversity of training data play a vital role in further imporving the performance of chat language model. </span>
다시 말해, 높은 quality 와 더 다양한 data 가 더 좋은 결과를 이끌어낼 수 있다는 것이다.
저자들은 million-scale multi-turn instructional conversation data 인 UltraChat 을 공개한다.
QA 나 Summarization 같은 task 를 활용하여 conversation 을 구성하지 않고, <span style='background-color: #dcffe4'>  Questions about the World, Creation and Writing, Assistance on Existing Material 이라는 세 sector를 curate 한다. </span>
이후 realistic multi-run conversation 을 구성하기 위하여, 두 독립적인 ChatGPT Turbo API 에게 대화를 진행시켜 query 와 response 를 생성하게 시킨다.

<span style='color:green;font-weight:bold'> ▶ Experiment </span>
<br>
LLaMA-13B 모델에 UltraChat 을 학습시켜 UltraLM 을 만들었다.
UltraLM 은 GPT-4 로 평가되었을 때 가장 높은 점수를 기록하였으며, 모든 open-source model 을 능가하는 퍼포먼스를 보인다.

## 2. Related Work
<span style='color:green;font-weight:bold'> Instruction Tuning </span>
<br>
[FLAN-T5](https://arxiv.org/pdf/2109.01652.pdf) 가 60 개의 NLP dataset 을 학습하여, LM 이 instruction tuning 을 통해 instruction following 능력을 갖출 수 있음을 보인 뒤, 많은 모델들이 instruction tuning 을 통해 학습되었다.
[T0](https://arxiv.org/pdf/2110.08207.pdf) 와 [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) 가 대표적인 예시이고, [FLAN2022](https://arxiv.org/pdf/2301.13688.pdf) 에서는 다양한 task 를 배우는 것이 out-of-distribution 일반화 성능이 좋음을 보였다.
InstructGPT 이후에는 강화학습을 이용한 human preference 학습에 대해서도 많은 연구가 있어왔다.

<span style='color:green;font-weight:bold'> Data Augmentation with LLMs </span>
<br>
Large-scale human-annotated instruction 을 모으는 것은 매우 어려운 일이다.
이를 위해 ChatGPT 나 GPT-3.5 와 같은 well-tuned LLM 으로 부터 sampling 한 data 를 모으는 것이 주목받고 있다.
예를 들어, Self-Instruct 나 Alpaca 는 Text-Davinci-003 을 distilling 하여 high-quality 의 instruction-reponse pair 를 생성한다.
Alpaca 의 성공은 LLM 에 data augmentation 을 부추겼다. 
그 성공으로, code-alpaca, alpacacot, GPT4ALL, ShareGPT, Dolly-v2, BELLE, Vicuna, Koala, Baize 등이 탄생하였다.
CAMEL 의 경우, multi-agent role-play 환경을 통해 real human conversation 을 simulate 한다.

## 3. Data Construction

## 4. Data Analysis

## 5. Experiments

## 6. Conclusion


<span style='color:green;font-weight:bold'> 초록색볼드체 </span>
<br>
<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
