---
layout: post
title:  "[ICML2022] Data Determinces Distributional Robustness in Contrastive Language-Image Pre-training (CLIP)"
date:   2022-11-14 16:30:00 +0900
use_math: true
categories: [Transformer, Vision-and-Language]
---
[[pdf]](https://proceedings.mlr.press/v162/fang22a/fang22a.pdf)  &emsp;

**Alex Fang<sup>1</sup>, Gabriel Ilharco<sup>1</sup>, Mitchell Wortsman<sup>1</sup>, Yuhao Wan<sup>1</sup>, Vaishaal Shankar<sup>2</sup>, Achal Dave<sup>2</sup>, Ludwig Schmidt<sup>1 3</sup>**
<br><sup>1</sup>University of Washington ,<sup>2</sup> Amazon, <sup>3</sup> Allen Institute for Artificial Intelligence. &emsp; 

![image](https://user-images.githubusercontent.com/42200027/201530852-d42832d4-ee65-47d1-92bd-e127c1648c0a.png)

# Abstract
- (Motivation) Pre-trained Language Model (PLM) 이 NLP task 를 푸는 굉장히 강력한 standard 가 되었지만, train 하기에는 computation cost 가 너무 비싸다. 
- (Solution) 이 연구에서는 simple and efficient learning framework **TLM** 을 제안하여, large-scale pretraining 에 rely 하지 않는 학습 방법을 제안한다. 
- (Method) Labeled task data 와 large general corpus 에 대하여, TLM 은 task data 를 Query 로 하여 general corpus 로부터 tiny subset 을 retrieval 한 후, task objective 를 jointly optimize 한다.  
- (Result) 4개 domain 의 8개 데이터셋에 대한 실험 결과, TLM 은 PLM 과 비교하여 FLOP 은 두 자리수나 적으면서 성능은 더 좋거나 유사한 성능을 보인다. 

# Introduction
