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
- (Motivation) CLIP, ALIGN, BASIC 과 같은 contrastive learning 기반의 vision-language model 들은 distribution shift 에 굉장한 robustness 를 보인다. 이렇게 큰 robustness gain 을 얻는 원인에 대한 질문은 굉장히 중요하다.
- (Solution) 체계적인 실험 조사(systematic experimental investigation) 으로 이 질문에 대해 탐구한다. 
- (Method) (1) Training set size (2) Training distribution (3) Language supervision at training time (4) Language supervision at test time (5) contrastive loss function 다섯 가지 possible cause 에 대해서 실험 조사를 진행한다.
- (Result) (2) Training distribution 이 다양할 수록 robustness gain 이 컸고, 나머지 네 개의 factor 들은 전혀 robustness 에 관련이 없었다.
- (New Dataset) Flickr annotation 으로  이뤄진 ImageNet version 의 새로운 dataset 인  **ImageNet-Captions** 을 공개한다. 이 데이터셋은 controllable vision-and-language training 이 가능하게 한다.

# Introduction
