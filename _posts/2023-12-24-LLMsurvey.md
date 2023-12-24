---
layout: post
title:  "A Survey of Large Language Models"
date:   2023-12-24 10:45:00 +0900
use_math: true
categories: [Transformer, PLM, LLM]
---

[[pdf]](https://arxiv.org/pdf/2303.18223.pdf)

**Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie and Ji-Rong Wen**

<span style='background-color: #dcffe4'> ㅁ 이 글은 Large Language Model (LLM) 의 survey 논문으로 cited paper 의 link 는 생략한다. </span>


# Abstract
  
- LLM:Large Langauge Model 은 tens or hundreds of billions of params 를 가지는 언어모델로, in-context learning 등의 몇몇 special ability 를 보인다는 측면에서 PLM:Pre-trained Langauge Model 과 차이를 보인다.
- 이 연구에서는 최신 LLM 연구를 **pre-training, adaptation tuning, utilization, capacity evaluation** 네 가지 측면에서 조사한 survey 논문이다.
 
# Introduction

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/e11e045f-e2b7-4622-bbba-23326484827c)


Machine 에게 있어 인간이 comminucate 하는 것과 유사하게, read, write 하는 기술을 갖게하는 것은 오랜 목표이다.

Langauge Modeling (LM) 은 machine 에게 언어 지능을 가르치는 major 한 방법으로, 다음 단어의 확률을 예측하도록 generative likelihood 를 model 하도록 학습시킨다.
LM 의 연구분야는 시대에 따라 크게 네 가지로 나뉜다.
- Statistical Language Models (SLM) : n-gram 기반으로 markov assumption 으로 word prediction 을 진행하는 통계적 방식이다. curse of dimension 문제가 발생한다.
- Neural Language Models (NLM) : MLP:multi-layer perceptron 이나 RNN:Recurrent Neural Network 등을 활용하여 word prob 을 예측한다. **NLP 연구의 매우 중요한 impact 를 가져온 연구들**이다.
- Pre-trained Language Models (PLM) : "pre-training" and "fine-tuning" 패러다임. ELMO, BERT, GPT-2, BART 등.
- Large Langauge Models (LLM) : [Scaling Law](https://arxiv.org/abs/2001.08361) 논문을 기반으로 PLM 의 성능이 scale 이 커짐에 따라 좋아진다는 연구가 있었다. 175-B GPT-3, 540-B PaLM 등이 그것인데, 이들은 성능이 그저 좋아지는 점을 넘어, complex task 를 푸는 special ability 를 보인다 (*in-context learning 등*)

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/070d3e95-d14a-49fb-b757-2ac501d835fc)

특히나, 위의 그림에서 보는 것처럼 <span style='color:green;font-weight:bold'> chatGPT </span> 의 등장 이후 LLM 연구가 매우 활발하다.
LLM 연구는 기존의 text data 를 model 하고 generate 하는 연구와 다르게, <span style='background-color: #dcffe4'>  complext task solving 을 하는데 치중되어 있다. (From langauge modeling to task sloving) </span>



<span style='background-color: #dcffe4'> 초록 배경색 </span>
<span style='color:green;font-weight:bold'> 초록 볼드체 </span>
