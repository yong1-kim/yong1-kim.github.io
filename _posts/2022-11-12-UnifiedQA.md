---
layout: post
title:  "[ICML2022] Describing Differences between Text Distributions with Natural Language"
date:   2022-11-12 17:38:00 +0900
use_math: true
categories: [LLM, Transformer]
---
[[pdf]](https://proceedings.mlr.press/v162/zhong22a/zhong22a.pdf)  &emsp;
[[github]](https://github.com/ruiqi-zhong/DescribeDistributionalDifferences) <br>

**Ruiqi Zhong<sup>1</sup>, Charlie Snell<sup>1</sup>, Dan Klein<sup>1</sup>, Jacob Steinhardt<sup>1</sup>**
<br><sup>1</sup> Computer Science Division, University of California, Berkeley. Correspondence to: Ruiqi Zhong  &emsp; 

![image](https://user-images.githubusercontent.com/42200027/201470594-acfa0565-f6df-44a6-88d6-4d92c3be2f60.png)

# Abstract
- (Motivation) 두 text 의 *distribution* 이 다르다는 것을 어떻게 알 수 있을까? 인간은 많은 sample 을 직접 읽는 과정이 필요하므로 굉장히 많은 시간이 걸린다. 
- (Solution) 이 논문에서는 **GPT-3 를 활용하여 automatically describe distribution of text** 방법을 제안한다. 이 방법은 기존에 없던 새로운 framework 로 다른 다양한 task 에도 적용 가능하다. 
- (Method) "[samples of $D_0$] + [samples of $D_1$] + the difference between them is __." 의 prompt 를 이용하여 GPT-3를 fine-tuning 시킨 뒤 
-  
# Introduction
