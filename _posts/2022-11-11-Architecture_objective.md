---
layout: post
title:  "[ICML2022] What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?"
date:   2022-11-11 15:48:00 +0900
use_math: true
categories: [LLM, Transformer]
---
[[pdf]](https://proceedings.mlr.press/v162/wang22u/wang22u.pdf)  &emsp;
[[github]](https://github.com/bigscience-workshop/architecture-objective) <br>

**Thomas Wang<sup>* 1</sup>, Adam Roberts<sup>* 2</sup>, Daniel Hesslow<sup>3</sup>, Teven Le Scao<sup>1</sup>, Hyung Won Chung<sup>2</sup>, Iz Beltagy<sup>4</sup>, Julien Launay<sup>3 5</sup>, Colin Raffel<sup>1</sup>**
<br><sup>*</sup> Equal Contribution, <sup>1</sup> Hugging face, <sup>2</sup> Google, <sup>3</sup> LightOn, <sup>4</sup> Allen institute for AI, <sup>5</sup> LPENS, Ecole Normale Superieure.   &emsp; 

![image](https://user-images.githubusercontent.com/42200027/201280889-750c58c2-b68b-4b58-9f82-2feb39f72b08.png)

# Abstract
- (Motivation) Large Language Model (LLM) 이 zero-shot generalization 에 좋은 성능을 보이지만, 많은 state-of-the-art 모델들이 각기 다른 architecture 와 pre-training objective 를 통해 학습되는데, 이 요소들에 대한 체계적인 비교(systematic comparison) 이 적다. 
- (Solution) 이 논문에서는 여러 LLM 들의 modeling choice 와 zero-shot generalization 에 대한 영향력을 평가하는 large-scale evaluation 방법을 제안한다.
- (Experiment) (causal decoder-only / non-causal decoder-only / encoder-decoder) 세 구조 (architecture)와 (autoregressive, masked language modeling) 두 가지 pre-training objective, 그리고 with/without multitask finetuning 조합들을 실험을 진행한다.
- (Results) causal decoder-only + autoregressive 방법이 zero-shot 성능은 제일 좋았으나, non-causal + mlm + multi-task finetuning 방법이 실험 성능은 제일 좋았다.
 
# Introduction
* 위의 그림의 module 블록들의 색과 아래의 색을 연결

Large Language Model(LLMs) 들은 unstructured text data 에 pre-train 된 후, additional training or labeled data 없이 다양한 방면의 task 에서 좋은 성능을 보인다. 이러한 능력을 <span style='background-color: #a5e5f2'> zero-shot generalization </span> 이라고 한다. 현재 대부분의 LLM 들은 <span style='background-color: #f7efb0'> [Transformer](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) architecure </span>  기반으로 구성되어 있다. Original Transformer 는 encoder-decoder 구조로 되어있지만, 많은 최근 LLM 들은 <span style='background-color: #f7efb0'> causal decoder-only </span> 모델로, auto-regressive 방법으로 학습한다([[1]](https://arxiv.org/pdf/1801.10198.pdf), [[2]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [[3]](https://ojs.aaai.org/index.php/AAAI/article/view/4182)) 그러나, [T5 model](https://arxiv.org/pdf/1910.10683.pdf) 에서는 <span style='background-color: #f7efb0'> Encoder-Decoder (ED) </span> model 을 통해 transfer learning 으로 decoder-only LLM 을 outperform 한다. 추가적으로, [UniLM](https://arxiv.org/pdf/1905.03197.pdf) 등의 <span style='background-color: #f7efb0'> Non-causal decoders-only </span> 구조는 attnetion mask 를 활용하여 decoder-only 와 encoder-decoder model 의 구조 사이의 gap 을 줄인다.
