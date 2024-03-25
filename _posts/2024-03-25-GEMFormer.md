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
<span style='color:green;font-weight:bold'> ▶Multi-Hop Question Answering(MHQA) </span>
<br>
Transformer 의 발전에 따라 정답을 추출하기 위해 여러 reasoning step 이 필요한 multi-hop question answering task 에 대한 연구가 활발하다.
MHQA 를 푸는 방법은 크게 두 가지로 나뉜다.
첫째는, sub-network 나 dedicated module 을 활용하여 supproting evidence 를 추출하여 활용하는 방법이다.
이 방법은 전적으로 evidence extraction 의 성능에 좌우되며, QA model 로 하여금 pre-selected factor 에 upper limit 이 되게된다.
둘째는, maxmimal input sequence length 를 크게하는 attention pattern 을 활용하여 long document encoding 을 활용하는 방법이다.
이 attention-based token representation 은 local information 과 global information 을 같은 vector 에서 처리하게 된다.
<span style='background-color: #ffdce0'> 이렇게 되면, high-level contextual feature 가 long seuqence 에 퍼지게 되어, 접근이 어려워진다. </span>

<span style='color:green;font-weight:bold'> ▶ GEMFormer (Global Explicit Memory Transformer) </span>
<br>
이 문제를 해결하기 위해 저자들은, **GEMFormer (Global Explicit Memory Transformer)** 를 제안한다.
<span style='background-color: #dcffe4'> 
GEMFormer 는 global information 을 저장하는 memory 를 활용하는 pre-trained language model augmenting 방법론이다. </span>
이것은 task 를 풀기 위해 중요한 정보가 담긴 memory sequence 에 long input 을 concat 하는 방법이다.
Token importance 는 language model uncertainty 로 정의된다.


## 2. Global Explicit Memory

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/f4fef6d5-4ddd-456a-8282-ed9276051956)

GEMFormer 는 RoBERTa 를 backbone 으로 활용한다.
<span style='background-color: #dcffe4'> Global explicit memory 는 정확한 reasoning 과 answer prediction 에 가장 중요한 document token 의 연속이다.  </span>
Model 의 uncertainty 가 input 의 중요도로 활용된다.
input sequence $x=[t_1,t_2,...,t_m]$ with $m$ tokens 에 대해, RoBERTa 의 LM head 를 통해 token probability vector $p=Softmax(LM-RoBERTa(x))$ 를 얻는다.
이후, Entory $H= -\frac{1}{n} \sum_{j=1}^n p_j log p_j$ 을 각각 input position 에 대해 구한다.
이 연구에서는 아래의 두 memory population condition 을 사용한다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/4aefbaab-1f9e-4e42-8c62-92e20eb2df7d)

$\theta$ 는 threshold 이고, $k$ 는 memory size 이다.

모델에 question 과 context 가 입력이 되면, 각각 contextual token 에 대한 entropy 가 결정된다.
이 entropy 는 question 과 token-surrounding context 에의해 결정된다.(conditional 하다)
Document 가 question-relevant collection 이라면, task-relevant token 의 entorpy 는 irrelevant one 보다 낮아야만 한다.

GEMFormer architecture 는 위의 그림과 같다.
RoBERTa 의 maximum sequence length limit 을 맞추기 위해, contextual document 는 여러 segment 로 나뉜 후, question 이 concat 된다.
Input processing 은 두 가지로 구성되는데 (1) document comprehension and memory population 과 (2) task-prediction generation 이다.
첫 번째 stage 에서, question-context segment 가 RoBERTa model 에 input 으로 들어간 후, LM head 에 의해 entropy 가 계산된다.
이후 위의 식 (1) 에 해당하는 entropy condition 을 만족하는 token 들이 선택되어 Global Memory (GM) 로 구성된다.
이후 두 번째 stage 에서는 question 과 globabl memory token 이 concate 되어 MHQA task training 에 사용된다.

실험은 세 영어 MHQA dataset : HotpotQA, 2WikiMultiHopQA, MusiQue-Ans 에 대해 진행된다.
각각은 HP, 2W, MSQ 라고 지칭한다.

## 3. Results and Discussion
<span style='color:green;font-weight:bold'> Main Results </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/92a5e305-2ffa-4230-b64c-a6822c1765a7)

- <span style='background-color: #dcffe4'> Low entropy 의 token 들이 Global Memory 로 활용될 때 좋은 성능을 보인다. </span>

<span style='color:green;font-weight:bold'> Improving ChatGPT performance </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/9e9b6310-91e1-4079-8949-5bd651fef5bb)

- <span style='background-color: #dcffe4'> Question only 와 비교했을 때 Retrieved passage 가 있어야 좋은 성능을 보인다. </span>
- <span style='background-color: #dcffe4'> Retrieved Passage 에 Gloabl Memory 를 썼을 때는 오히려 성능이 안좋아졌지만 (Q+R > Q+M+R), Full context 를 썼을 때는 Global Memory 를 쓰는 게 더 좋다(Q+M+C > Q+C) </span>

<span style='color:green;font-weight:bold'> Ablation Study </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/3a4d7a4f-3bf9-4624-a08a-21add9f69959)

- <span style='background-color: #dcffe4'> Memory filling 을 위해서는 Question 이 필수불가결하고, Finetuning 은 도움이 되며, Rnadom memory 는 효과가 좋지 않다 </span>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/7dfc8306-b8de-40bd-a3b5-526a14a1ed14)

- <span style='background-color: #dcffe4'> 위의 그림처럼 학습과정에서 Supporting fact 의 entropy 가 낮아지기 때문에, random memory 를 활용하기보다는 low entropy rule 을 쓰는 것이 더 좋다는 것을 확인할 수 있다. </span>

<span style='color:green;font-weight:bold'> Memory Analysis </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/0b4bc356-09d9-4267-9d89-1476f365ca3f)

- <span style='background-color: #dcffe4'> Memory Size 가 크면 클 수록 좋다. </span>

## Conclusion
```
In this study, we demonstrated how utilizing uncertainty-based global explicit memory can enhance the model performance on MHQA tasks. Our findings indicate that utilizing low entropy context tokens can aid the model in MHQA reasoning, but only when the entropy estimation model is specifically fine-tuned to the target task. Experiments show that higher-performing models use larger memory sizes with better coverage of supporting facts.
```
## Limitations
```
There are several limitations to this work. First, the global explicit memory augmentation of the input sequence may increase the training time by shortening the context chunk lengths. Second, the current implementation of memory token selection results in storing a significant fraction of irrelevant tokens which interferes with the calculation of correct predictions. We will work on methods to improve the relevance of information stored in memory
```
