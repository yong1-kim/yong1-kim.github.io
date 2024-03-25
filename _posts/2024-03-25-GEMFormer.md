---
layout: post
title:  "[EMNLP2023] Uncertainty Guided Global Memory Improves Multi-Hop Question"
date:   2024-03-25 22:00:00 +0900
use_math: true
categories: [LLM, PLM]
---

[[pdf]](https://aclanthology.org/2023.emnlp-main.262.pdf) &emsp;
[[github] ](https://github.com/Aloriosa/GEMFormer)

**Alsu Sagirova<sup>1</sup>, Mikhail Burtsev<sup>2</sup>**
<br><sup>1</sup> Moscow Institute of Physics and Technology, Dolgoprudny, Russia <sup>2</sup> London Institute for Mathematical Sciences, London, UK &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/c0b633a6-77cf-4085-9169-a7674ec575d7)

## Abstract
- (**Multi-Hop QA**) Multi-hop QA 는 두 가지 접근 방법이 많이 사용되는데, 첫째는 여러 supporting evidence 를 찾아내는 것이고, 둘째는 attention mechanism 을 활용하여 long input encoding 을 facilitate 하는 것이다.
- (**Lack of global attention**) 그 중 attention-based 접근 방법은 reasoning step 을 연결해주는 explicit global contextual information 이 부족하다.
- (<span style='color:green;font-weight:bold'> GEMFormer </span>) 저자들은, (1) entire document 에서 relevant information 을 찾아 memory 에 저장하고 (2) 그것들을 local context 에 결합하는 two-step approach 인 **GEMFormer** 를 제안한다.
- (**Experiment**) memory-augmented input 과 함께 pre-trained model 을 finetuning 한 결과, 세 multihop QA dataset 에서 baseline 대비 향상을 이룬다. 추가적으로, global explicit memory 가 정확한 answer 를 위해 필요한 supporting fact 를 잘 담아내는 것을 확인한다.

## 1. Introduction

## 2. Global Explicit Memory

## 3. Results and Discussion

## Conclusion
```
In this study, we demonstrated how utilizing uncertainty-based global explicit memory can enhance the model performance on MHQA tasks. Our findings indicate that utilizing low entropy context tokens can aid the model in MHQA reasoning, but only when the entropy estimation model is specifically fine-tuned to the target task. Experiments show that higher-performing models use larger memory sizes with better coverage of supporting facts.
```
## Limitations
```
There are several limitations to this work. First, the global explicit memory augmentation of the input sequence may increase the training time by shortening the context chunk lengths. Second, the current implementation of memory token selection results in storing a significant fraction of irrelevant tokens which interferes with the calculation of correct predictions. We will work on methods to improve the relevance of information stored in memory
```



<span style='color:green;font-weight:bold'> 초록색볼드체 </span>
<br>

<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
