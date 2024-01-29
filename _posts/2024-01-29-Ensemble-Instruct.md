---
layout: post
title:  "[EMNLP2023] Ensemble-Instruct: Generating Instruction-Tuning Data with a Heterogeneous Mixture of LMs"
date:   2024-01-29 12:45:00 +0900
use_math: true
categories: [LLM, PLM]
---

[[pdf]](https://aclanthology.org/2023.findings-emnlp.836.pdf) &emsp;
[[github]](https://github.com/IBM/ensemble-instruct)

**Young-Suk Lee, Md Arafat Sultan, Yousef El-Kurdi, Tahira Naseem, Asim Munawar, Radu Florian, Salim Roukos, Ramón Fernandez Astudill**
<br>IBM Rsearch AI &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/b4b6e01f-c0c8-4aad-b083-9932c1958196)

## Abstract
- (**Data Generation from ICL**) Self-Instruct 나 Alpaca 와 같이, ICL 을 활용하여 data 를 generation 하는 것을 통해, 적은 양의 human supervision 으로 모델을 학습시킬 수 있다.
- (<span style='background-color: #ffdce0'>Limitation</span>) 그러나 이러한 방법은 상표가 있거나 공개되지 않은 175B 정도 크기의 LLM 에 의존(resort) 할 수 밖에 없다.
- (**Proposed Method**) 이 논문에서는 permissive license 를 가지며, 10-40B 정도의 비교적 작은 모델을 가지고도 이러한 Technique 를 구현한다. 저자들은 이 정도 size 에서는 SELF-INSTRUCT 방법이 좋지 못함을 보임과 동시에, 새로운 ICL method 를 제안한다.
- (**(1) Categorization**) 첫 번째 idea 는 LM 이 학습하기 쉬운 ICL template 을 categorize 하고 simplify 하는 것이다.
- (**(2) Ensembling**) 두 번째 idea 는 여러 LM output 을 앙상블하여, high-quality synthetic example 을 생성하는 것이다.
- (**Experiment**) SELF-INSTRUCT 와 같은 세팅을 ㅗ실험한 결과, SELF-INSTRUCT 보다 더 좋은 퀄리티의 instruction 을 생성하고, 이를 통한 instruction tuning 으로 성능을 더 끌어올렸다.

## 1. Introduction
<span style='color:green;font-weight:bold'> ▶너무 큰 LLM 에 의존하는 기존 Instruction dataset generation via ICL </span>
<br>
Instruction-tuned LLM 은 정말 많은 일을 수행할 수 있다.
이를 위하여 Large-scale instruction-tuning data를 automatic 하게 synthesis 하는 연구가 활발히 진행되고 있다.
예를 들어, SELF-INSTRUCT 는 작은 크기의 expert-seed example 을 ICL(In-Context Learning) 을 통해 bootstrapping 하여 instruction-tuing dataset 을 생성한다.
이 방법은 매우 강력하며, 이를 통해 LLAMA 를 학습한 Stanford Alpaca 등의 follow-up 연구도 많지만, 이러한 것들은 여전히 175B 크기의 LLM 에 의존한다는 단점이 있다.

<span style='color:green;font-weight:bold'> ▶Ensemble-Instruct </span>
<br>
<span style='background-color: #dcffe4'> 이 논문에서는 fully accessible 한 40B 정도의 smaller LM 을 통한 high-quality instruction tuning data generation 을 할 수 있는 Ensemble-Instruct 라는 방법론을 제안한다. </span>
우선, 저자들은 이 정도 크기의 작은 모델에는 SELF-INSTRUCT 방법이 성능이 좋지 못함을 보이고, (1) Categorizating and simplifying the ICL propmt 와 (2) Ensembling over multiple LM output 의 두 가지 방법을 main idea 로 하는 Ensemble-Instruct 방법론을 제안한다.

조금 더 자세하게는, SELF-INSTRUCT 방법이 *instruction* 을 생성한 후, *input first* 와 *output first* 을 통해 instance 를 생성하는 반면, Ensemble-Instruct 는 *without input* 과 *with input* 으로 categorizing 하고 이를 위한 prompt 르f simplifying 한다. 이후, heterogenous collection 들을 모은 뒤, majority voting 을 통한 ensemble 방법을 적용한다.

<span style='color:green;font-weight:bold'> ▶Experiment </span>
<br>
작은 모델로 사용되는 모델들은 T5 UL2-20B, FALCON-40B, FLAN-T5-11B, FLAN-UL2-20B, GPT-NeoxX-20B(chat-tuned) 등이다.
추후, instruction tuning 을 진행하는 base model 은 Pythia-1.4B 와 MPT-7B (decoder only LM similar to LLaMA), GPT-JT-6B (instructed version of GPT-J) 등이다. 
언급된 모든 모델들은 open-source 이며, permissive license (Apache-2)를 갖고 있다.

SELF-INSTRUCT 와 유사하게 SUPERNI 에 test 해 본 결과, 좋은 성능을 보였으며, 생성한 synthetic instruction-tuning dataset 을 release 하였다.

## 2. Ensemble-Instruct

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/fbee7404-8cc0-429a-b8c7-7ecd1fa48830)

Ensemble-Instruct 의 overview는 위의 figure 와 같다.
이는 세 가지 main component 로 이뤄져있다: (1) Categorization of tasks and their prompts, (2) Generation of instructions followed by instances (where an instance comprises an input and an output, (3) Ensemble of outputs from multiple LMs.

# 2.1. Categorization of Tasks and Prompts
저자들은 input 이 필요한 instruction (type-A) 과 input 이 필요하지 않은 instruction (type-B) 로 type 을 나눈다.
아래의 figure 에서 그 예시들을 볼 수 있다. 
SELF-INSTRUCT 의 시작 175 seed set 을 구분하면, type-A 가 125개, type-B가 50개이다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/b2e52f2d-a13c-465b-a9d9-bcd237c58f36)

# 2.2. Instruction Generation
Type-A 를 위해서는 24개의 ICL exempler (demonstration) 을 사용하고, 이 때 20개는 125 개 시작 seed task 에서 추출하고, 4개는 앞서 생성된 instruction 에서 randomly sample 한다. 
Type-B 를 위해서는 10개의 ICL exempler 를 사용하고, 8개는 125 개 시작 seed task 에서, 2개는 이전에 생성된 instruction 에서 생성한다.

역시, SELF-INSTRUCT 를 따라서, ROUGE-L score 가 0.7 이하로 겹치는 것만 남기고 filtering out 한다.

# 2.3. Instacne Generation
Type-A 를 위해서 18 개의 ICL exempler 를, Type-B 를 위해서 15 개의 exempler 를 사용한다. 
위의 Figure 2 에서 Type-A 와 type-B 예시를 볼 수 있다.

# 2.4. Output Ensembling
**지금까지 setting 은 categorization 을 진행한 것 외에는, 사실상 SELF-INSTRUCT 와 크게 다를 것이 없다.**
하지만 smaller model 을 사용한 만큼 그 결과가 부정확할 확률이 매우 높다.
<span style='background-color: #dcffe4'> 따라서, additional set of LM 들의 output 을 앙상블하는 방법론 </span> 을 사용한다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/a54a8f32-05a3-41e6-b926-5a466d4ba249)

위의 알고리즘과 같이, ensemble 을 진행하는데, 우선 all three pair 를 ROUGE-L score 로 유사도를 측정한다.
만약, 모든 ROUGE-L score 가 threshold 를 넘는다면 (가장 낮은 score 가 thershold 를 넘는다면), **가장 높은 ROUGE-L pair 의 첫 번째 element 를 return** 한다.
저자들은 이 것이 Minimum Bayesian Risk decoding 의 greedy version 이라고 한다.

## 3. Analysis of Instruction Tuning Dataset

<span style='color:green;font-weight:bold'> 초록색볼드체 </span>

<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
