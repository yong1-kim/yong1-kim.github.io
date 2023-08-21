---
layout: post
title:  "[ACL2023] FutureTOD: Teaching Future Knowledge to Pre-trained Language Model for Task-Oriented Dialogue"
date:   2023-08-19 13:00:00 +0900
use_math: true
categories: [Dialogue, PLM]
---
[[pdf]](https://aclanthology.org/2023.acl-long.360.pdf) &emsp;
[[github]](https://github.com/Zeng-WH/FutureTOD)

**Weihao Zeng<sup>*1</sup>, Keqing He<sup>*2</sup>, Yejie Wang <sup>1</sup>, Chen Zeng <sup>1</sup>, Jingang Wang <sup>2</sup>, Yunsen Xian <sup>2</sup>, Weiran Xu<sup>*1</sup>**
<br><sup>1</sup> Beijing University of Posts and Telecommunications, Beijing China, <sup>2</sup> Meituan, Beijing, China  &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/187572dd-b8dd-4d80-b589-53ddd47ce6bc)

# Abstract
- (Motivation) PLM 이 NLP scenario 에서 큰 성공을 거두고 있지만, 일반적인 text 학습과 task-oriented dialog 학습의 intrinsical 차이로 practically less useful 하다.
- 최근의 dialog pretraining 방법은 contrastive framework 에 의존하지만, positive 와 hard negative 를 selecting 하는데 어려움을 겪고 있다.
- (FutureTOD) 이 논문에서는 previous dialog context 에서 **self-training** 기법을 활용하여 future knowledge 를 distil 하는 FutureTOD 를 제시한다.
- (Intution) 이 것은 좋은 dialog representation 은 local context information 을 학습함과 동시에 future info 를 predict 할 수 있어야 한다는데서 intuition 을 얻는다.
- FutureTOD 는 성능면에서 우수하고, 특히 generatlization 과 robustness 에서 우수하다.
   
# Introduction
