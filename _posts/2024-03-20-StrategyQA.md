---
layout: post
title:  "[TACL2021] Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies"
date:   2024-03-20 18:00:00 +0900
use_math: true
categories: [Retrieval, LLM, PLM]
---

[[pdf]](https://aclanthology.org/2021.tacl-1.21.pdf)  &emsp;
[[code and dataset]](https://allenai.org/data/strategyqa)

**Mor Geva<sup>1,2</sup>, Daniel Khashabi<sup>2</sup>, Elad Segal<sup>1</sup>, Tushar Khot<sup>2</sup>, Dan Roth<sup>3</sup>, Jonathan Berant<sup>1,2</sup>**
<br><sup>1</sup> Tel Aviv University <sup>2</sup> Allen Institute for AI <sup>3</sup> University of Pennsylvania &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/cb198cb6-b314-412b-9770-4585a63fb1fb)

# Abstract
- (**Explicit Multihop reasoning**) 현재 Multihop reasoning 의 큰 단한계점은 question 이 explicit 하다는 것이다.
- (<span style='color:green;font-weight:bold'> StrategyQA </span>) 이 논문에서는 question 속의 reasoning step 이 implicit 하게 내재되어 있는 StrategyQA 라는 QA benchmark 를 제시한다. 저자들은 term-based priming 기법을 통해 annotator 들로 하여금 창의적인 질문을 생성하게 하였고, adversarial filtering 과정을 거쳐 벤치마크를 생성하였다. 
- (**Statistics and Analysis**) 2,780 example 에 각각 decomposition 과 evidence paragrah 를 포함한다. StrategyQA 는 short, topic-diverse 하면서 넓은 범위의 strategy 를 cover 하고, 87% 점수의 human score 와 66% score 의 baseline score 를 report 한다.

## 1. Introduction

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/67d0c7ef-12ed-4d58-9206-ec9da2be94bb)

<span style='color:green;font-weight:bold'> ▶ Multi-hop Reasoning </span>
<br>
최근 input 을 해결하기 위한 여러 단계의 추론이 필요한 Multi-hop reasoning 에 대한 연구가 활발해지며, 모델과 벤치마크가 많이 제안되었다.
<span style='background-color: #ffdce0'> 하지만 기존의 벤치마크들의 question(query)은 보통 explicit 하게 주어진다.  </span>
예를 들어, 위의 그림에서처럼 기존의 데이터셋들은 _"Was Aristotle alive when the laptop was invented?"_ 와 같이 정보를 추출하기 위해 직접적으로 언급이 되게끔 질문이 구성된다.
그러나 real-life question 은 _"Did Aristotle use a laptop?"_과 같이 같은 step 으로 해결하지만, question 속에 정보가 implicit 하게 내재되어있는 경우가 많다.

<span style='color:green;font-weight:bold'> ▶ Challenge in Implicit Question </span>
<br>
**IMPLICIT** question 에 답변을 하는 것은 challenging 하다.
우선, question 과 context (evidence) 사이의 overlap 이 적기 때문에 정보를 retrieve 하기 어렵다.
게다가 question 의 길이가 보통 짧기 때문에, 이러한 측면에서 implicit question 은 lexcial overlap 등의 shortcut 을 찾을 확률이 적어진다.

<span style='color:green;font-weight:bold'> ▶ StrategyQA </span>
<br>
<span style='background-color: #dcffe4'> 
이에 저자들은 *strategy question* 으로 구성된 implicit multi-hop reasoning 을 위한 Boolean QA bnechmark 인 StratgyQA 를 제시한다.
 </span>
여기서 말하는 *strategy* 는 question 으로부터 atomic sub-question 을 추출할 수 있는 능력을 말한다. (=decomposition ability)

Strategy question 을 구성하기 위해 crowdsourcing 을 활용하는 것은 쉬운 일이 아니다.
Annotator 에게 창의성(creativity)를 요구해야 하는 과정이므로, 기존 벤치마크에서 entire context 를 보여주고 multi-hop question 을 생성시키는 것을 넘어서야 한다.
이에 저자들은 다음의 과정들로 dataset construction pipeline을 구성한다.
(a) Annotator 로 하여금 imagination 과 creativity 향상을 위해 최소한의 context 를 부여하고, (b) diversity 를 위해 최대한 많은 annotator 를 고용하였으며 (c) adversarial model 학습을 통해, data collection 과정에서 recurring pattern 을 방지하고 난도를 증가시키는 과정을 거친다.

<span style='background-color: #dcffe4'> StrategyQA 는 question decomposition 과 evidence paragraph 를 포함한다. </span>
위의 Figure 에서 "D" 처럼 question 을 sub-question 으로 나누는 decomposition 정보와 그 것에 해당하는 Evidence ("E") 로 구성되어있다.
벤치마크 분석에서 StrategyQA 는 physics, geography 등 다양한 knowledge domain 에 걸쳐있으며, retrieval 과 QA 모두에서 challenging 함을 드러낸다.

## 2. Strategy Qeustions

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/332fe333-dd97-4034-8b82-fa38c76387e5)

# 2.1. Desiderata
QA 벤치마크 생성에는 여러 desired criteria 가 존재할 수 있다. Answerable 에 대한 연구도 많이 진행되고 있고, Hallucination 에 대한 연구도 많이 진행되고 있기 때문에 이러한 needs 에 따라 데이터셋이 존재할 수 있다. StrategyQA 에서는 이러한 측면보다는 implicit query 구성에 desiderata 를 맞춘다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/eddf9cc6-db70-49b3-90fe-131a425f5acc)

- **(1) Multi-Step** : 첫 번째 figure 처럼 여러 개의 질문으로 구성되어 있으며, 각 질문의 답변을 통해 logical operation 까지 할 수 있어야 한다.
- **(2) Feasible(Answerable)** : Question 은 corpus 속의 paragraph 로부터 answerable 해야 한다. 
- <span style='color:green;font-weight:bold'> (3) Implicit </span> : Key property 이며, questino 의 자연어 (natural language) 그대로 쉽게 정보를 추출하기 힘들어야 한다.
- **(4) Definite** : 명확한 대답을 할 수 있어야 한다. 예를 들어, "나무에 전기가 통하는가?" 라는 질문에 대해 어떠한 나무는 잘 통할 수 있지만 (환경에 따라) 정답은 generally "no" 인 것처럼 명확한 대답을 할 수 있어야 하고, "햄버거를 샌드위치라 볼 수 있는가?" 같은 답변이 갈릴 수 있는 질문은 하지 않는다.
- **(5) Boolean**: 모든 대답은 yes or no 이다.
- 
<span style='background-color: #dcffe4'> 논문에서 말하는 Implicity 의 정의 : </span>
```
a precise definition of implicit questions based on lexical overlap is elusive, but a good rule-of-thumb is the following:
If the question decomposition can be written with a vocabulary limited to words from the questions, their inflections, and function words, then it is an explicit question.
```

# 2.2. Decomposing Strategy Questions

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/12ea1eff-b297-47d1-87a0-52a645c2d235)

저자들은 위의 desiderata 에 따라 모든 question 이 decomposition 할 수 있게 annotate 한다.
<span style='background-color: #ffdce0'> 기존의 방법들은 대부분 rationale 혹은 supporting fact 라고 불리는 작은 text snippet 으로 decomposition 이 구성되었지만, 저자들은 진정한 reasoning 은 context 속에 explicit 하게 등장하지 않는다고 주장한다.
 </span>
 이에 저자들은 모든 question-answer pair 에 **strategy question decomposition** 을 적용한다.
모든 question $q$ 는 $n$ 개의 step $<s_1, s_2, ..., s_n>$ 으로 구성되고, 각각의 step $s_i$ 는 single-step question 이며 각각 spcial reference 를 포함한다.
이 special reference 는 직전 step 의 결과를 refer 하는 placeholder 이다.
마지막 decomposition 인 $s_n$ 은 final answer 를 return 하는 step 이다.

위의 Table3 에서 decomposition step 을 볼 수 있다.
첫번째 row의 explicit question ([[QDMR]](https://aclanthology.org/2020.tacl-1.13.pdf)) 은 decomposition 이 small vocab 으로 제한된다.
그러나 나머지 세 row 의 implicit decomposition 을 vocab 에 제한 없이 어떠한 token 도 등장할 수 있으며 각각은 implicit reasoning 을 구성하기만 하면 된다.
각각의 decomposition step 은 retrieval step 과 operation step 으로 나뉘는데, 맨 처음 figure 나 위의 두 번째 row 처럼 일단 정보를 추출해오는게 retrieval step 이고, 추출된 정보에서 logical inference 를 하는 것이 operation step 이다.


## 3. Data Collection Pipeline
※ 논문참고

## 4. The STRATEGYQA Dataset
Dataset 구성을 위해 29 question writers, 19 decomposers, 54 evidence matchers 를 고용하였다.
2,835 개의 question 을 collect 했으며, 그 중 55개는 filter 되어 2,780 개의 question 으로 구성된다.

# 4.1. Dataset Statistics
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/5915fac9-518d-4cce-b2de-30f56a4adb83)

# 4.2. Data Quality
<span style='color:green;font-weight:bold'> Do questions in STRATEGYQA require multistep implicit reasoning? </span>
<br>
Quality 측정의 위해 100 random sample 을 조사하였다.
두 expert (=author) 가 조사하였다고 한다. 그 결과, 대부분(81%)이 valid multi-step implicit question 이었다고 한다.

<span style='color:green;font-weight:bold'> Do questions in STRATEGYQA have a definitive answer? </span>
<br>
Expert 가 web 에 접근할 수 있는 환경에서 question 들을 분석한 결과, 94% 의 question 이 agree 하고 단 2% 에서만 disagree 했다고 한다. 나머지 4% 는 abimgiuous 했다.

<span style='color:green;font-weight:bold'> What is the quality of the decompositions? </span>
<br>
Expert 는 decomposition 이 잘되었는지, 그리고 그것들이 explicit or implicit 한지 분석하였다.
그 결과, 83% decomposition 이 valide 하게 sub-question 으로 break-down 되었으며, 17% 는 explicit 하게 decomposition 되었다고 한다.
그러나 그 17% 중 14% 는 이미 original question 자체가 explicit 하다고 한다.


<span style='color:green;font-weight:bold'> Would different annotators use the same decomposition strategy? </span>
<br>
50 개의 sample 을 뽑은 뒤, 다른 worker 들에게 question 을 decompose 하게 시켰다. 그 결과, 44개 (88%) 에서 같은 reasoning path 를 보였다. 이 결과는 다른 worker 들을 활용해도 decomposing 과정에 같은 strategy 를 사용한다는 것을 보여준다.

<span style='color:green;font-weight:bold'> Is the evidence for strategy questions in Wikipedia? </span>
<br>
각각의 decomposed question 들이 evidence 와 matching 되는지 세 worker 가 매겼을 떄, 88.3% 의 대부분의 question 이 fully coverd 되었고, 86.9% question 이 최소한 하나의 worker 에게 evidence 와 match 된다고 한다.

# 4.3. Data Diversity

<span style='color:green;font-weight:bold'> (1) Reasoning Skills </span>
<br>

- Strategy Diversity

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/e7599889-9075-487c-810f-5aab05d97479)

- Domain-related and logical reasoning skill diversity

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/ee80d991-a45a-4ea6-924e-a27f7a00e775)


<span style='color:green;font-weight:bold'> (2) Question Topics </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/1262b529-0762-4cc2-889e-4e88f798052f)

# 4.4. Human Performance
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/ada62c86-c082-417a-a102-1dcbcffb39aa)

100개의 sample 을 뽑아 expert (=author)가 question 에 대답을 해본 결과이다.
87% 정도의 정답률을 보이고, error analysis 에서 main reason to failure 는 evidence 를 찾기 힘들 때이다.

## 5. Experimental Evaluation
세 가지 측면에서 벤치마크를 분석한다.
- a) LM 이 strategyQA 를 잘 푸는가?
- b) relevent context 를 retreival 하는 것이 helpful 한가?
- c) decomposition 이 도움이 되는가?

# 5.1. Baseline Models
- Backbone model : BOOLQ, MNLI, TWENTY QUESTION, DROP 으로 finetuned 된 ROBERTA
- Setting : No context / With context (by BM25 Retrieval)
- Predicting Decompositions : BART 를 학습시켜, question 을 decomposition
- Baseline dmodels : ROBERTa - No retrieval / REBERTA : retrieval with gold decomposition / ROBERTA gold decomp and gold paragraph
  
# 5.2. Results

<span style='color:green;font-weight:bold'> Strategy QA performance </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/fdfa79b6-b93e-4ee6-b6af-d9b634fa91e5)

- Maximum score : ACC 72.0

<span style='color:green;font-weight:bold'> Retrieval Evaluation </span>
<br>

- Maximum score : Recall@10 0.282

## Concclusion
```
We present STRATEGYQA, the first dataset of implicit multi-step questions requiring a widerange of reasoning skills. To build STRATEGYQA, we introduced a novel annotation pipeline for eliciting creative questions that use simple language, but cover a challenging range of diverse strategies. Questions in STRATEGYQA are annotated with decomposition into reasoning steps and evidence paragraphs, to guide the ongoing research towards addressing implicit multi-hop reasoning.
```
