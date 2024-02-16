---
layout: post
title:  "[EMNLP 2023] Poisoning Retrieval Corpora in Injecting Adversarial Passages"
date:   2024-02-16 22:35:00 +0900
use_math: true
categories: [Retrieval, LLM, PLM]
---

[[pdf]](https://aclanthology.org/2023.emnlp-main.849.pdf) &emsp;
[[github]](https://github.com/princeton-nlp/corpus-poisoning)

**Zexuan Zhong<sup>†∗</sup>, Ziqing Huang<sup>‡∗</sup>, Alexander Wettig<sup>†</sup>, Danqi Chen<sup>†</sup>**
<br><sup>†</sup>Princeton University <sup>‡</sup> Tsinghua University &emsp;


![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/23ca3f43-e14a-4f19-b32b-eb4494cb5fe8)


## Abstract
- (**Adversarial Attacks on Retrieval system**) 이 논문에서는 dense retreival 에 discrete token 을 purtbing 하여 training query set 에서 similarity 를 maximize 하는 adverarial passage 를 집어넣는 adversarial attack 방법을 소개한다. 이러한 adversarial passage 가 주입되면, retrieval system 을 fooling 하는데 큰 효과가 있다.
- (**Generalization**) 심지어, 이 방법은 out-of-domain 에 대한 일반화 성능까지 가지는데, Natural Question 에 대해 optimize 된 adversarial attack 이, finnancial domain 이나 online forum 에서도 94% 이상의 attack 성능을 보인다.
- (**Experiment**) 다양한 dense retriever 에 attack 을 진행하여 benchmark 를 세웠을 때, 대부분의 retriever 가 500 passage 정도면 attack 에 취약함을 보였고, 이 500 passage 는 보통 million 단위의 passage 를 갖는 corpus 크기에 비하면 굉장히 극소량이다.

## 1. Introduction

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/63f0fccd-6103-4892-9c65-cf6ef5bce445)

<span style='color:green;font-weight:bold'> ▶ What extent can retriever be safely? </span>
<br>
Dense Retriever 가 인공지능의 영역에 들어서면서, 기존의 lexical method 에 비해 훨씬 높은 성능을 보이고 있다.
그러나, 여전히 long-tail entity 에 대해서는 성능이 약하고, out-of-domain generalization 성능은 떨어져서, 실제 real-world scenario 에서의 extent 는 어느정도인지 의문이 든다.

<span style='color:green;font-weight:bold'> ▶ Corpus Poisoning Attack </span>
<br>
이 논문에서는 새로운 타입의 vulnerability 를 보인다.
이 것은 <span style='background-color: #dcffe4'> corpus poisoning attack 으로, small fraction 의 adversarial passage 를 주입하는 것으로 system 을 fooling 하는 것</span>이다.
<span style='background-color: #ffdce0'> 기존의 연구에서 individual query 에 대해 adversarial passage 가 craft 될 수 있음을 보인 것과 다르게, </span> 이 논문에서는 user query 의 broad set 으로 부터 생성되고, out-of-domain 에 generalization 성능까지 갖춘다.
이러한 세팅은 Wikipedia 나 Reddit 같은 online forum 에 현실적으로 적용가능하고, black hat SEO (Search Engine Optimization) 의 새로운 도구가 될 수 있다.

이 attack 은 **HtotFlip** method 에 영감을 받은 gradient-based method 이고, 이 것은 <span style='background-color: #dcffe4'> discrete token 을 iteratively perturb 하여, training query set 에서 similairy 를 maximize 하는 방법 </span> 이다.
또한 simple clustering 기법을 차용한다.

<span style='color:green;font-weight:bold'> ▶ Experiment </span>
<br>
다양한 state-of-the-art dense retreiver 에 제안된 attack 방법을 적용하였을 때, 아주 미량의 adversarial passage 만으로 system 을 바보로 만든다.
특히, unsupervised Contriever model 이 취약한데, 10개 adversarial passage 만으로, 90% 의 query 를 속일 수 있다.
Supervised retriever 인 DPR, ANCE 등의 경우는 공격이 조금 힘들지만, 500 passage 정도 만으로도 50% 의 공격 성공율을 얻을 수 있다.
또한,  single-token change 에 sensitive 하지 않은 multi-vector retriever 인 ColBERT 등에도 효과가 좋은 공격 방법이다.
마지막으로, 이 방법론은 out-of-domain retrieval corpora 에서도 일반화 성능이 좋다.

## 2. Method
# 2.1. Problem Definition
Dense Retriever 는 dual encoder 로, passage encoder $E_p$ 와 query encoder $E_q$ 에 대하여, inner product 를 embedding similarity 로 활용한다 : $sim(q,p) = E_q(q)^{T} E_p(p)$.

Supervised setting 인 DPR, ANCE 나 unsupervised setting 인 Contriever 에서 모두, 이 dual encoder 들은 contrastive learning objective 로 학습된다.
Inference 할 때는, nearest-neighbor clustering 을 사용한다.

# 2.2 Corpus Poisoning Attack
한 번 corpus 가 poison 되면, (Once the corpus has been poisoned), dense retriever 는 adversaril passage 를 retrieve 할 걸로 기대된다.
Adversarial passage $A= \{a_1, a_2, ..., a_{|A|}\}$ ($|A|<< |C|$$)에 대해, at least 하나의 adversarial passage 가 top-k nearest cluster 에 포함되는 것이 objective 이다.

# 2.3. Optimization
Query set $|Q|$ 에 대하여, 가장 retrival resul 에 많이 포함될 수 있는 adversarial passage $A$ 를 얻는 것이 목표다.
Model 을 mislead 하기 위해, 아래의 수식을 통해 sequence of token $a=[t_1, t_2, ...]$ 를 찾는다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/9e9edf35-7480-481c-a827-15aea4283edf)

우선, single passage 를 어떻게 generate 하는지 살펴보고, multiple passage 로 넘어가면, HotFLIP 에 영향을 받아 optimization problem 을 푸는 gradient-based method 를 제안한다.
이 방법론은 <span style='background-color: #dcffe4'> token 을 replace 하면서, model 의 output 을 approximate 하는 방법이다. </span>
우선, adversarial passage 로 random passage 를 시작점으로 한다. 
각각의 step 마다 token $t_i$ 가 다른 token $t_i`$ 로 바뀔 때의 model output 의 approximation 을 계산한다.
이 approximation 계산을 HotFlip 과 같이, gradient 를 사용하고 , 그 수식은 ![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/ba90ec95-4a93-40fc-818c-b950d79ec4d4) 이다.


vec2text 와 연계하면 하나의 연구가 뚝딱일 수도?







<span style='color:green;font-weight:bold'> 초록색볼드체 </span>

<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
