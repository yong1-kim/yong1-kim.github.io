---
layout: post
title:  "[ACL2023] SELF-INSTRUCT: Aligning Lnaugage Models with Self-Generated Insructions"
date:   2024-01-26 17:00:00 +0900
use_math: true
categories: [LLM, PLM]
---

[[pdf]](https://arxiv.org/pdf/2212.10560.pdf) &emsp;
[[github]](https://github.com/yizhongw/self-instruct)

**Yizhong Wang <sup>♣</sup>, Yeganeh Kordi <sup>♢</sup>, Swaroop Mishra <sup>♡</sup>, Alisa Liu <sup>♣</sup> Noah A. Smith <sup>♣+</sup>, Daniel Khashabi <sup>♠</sup>, Hannaneh Hajishirzi <sup>♣+</sup>**
<br><sup>♣</sup> University of Washington <sup>♢</sup> Tehran Polytechnic <sup>♡</sup> Arizona State University <sup>♠</sup> Johns Hopkins University <sup>+</sup> Allen Institute for AI &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/16fe65d2-8530-4578-a620-797d64dcf87a)

## Abstract
- (**Instruction Tuning**) Instruction 에 respond 할 수 있게 language model 을 finetuning 하는 instruction tuning 을 통해 새로운 task 에 대한 높은 일반화 성능을 부여할 수 있다.
- (**Lack of dataset**) 그러나, human-written instruction data 는 그 양과 다양성, 창의성(creativity) 가 부족하다.
- (<span style='color:green;font-weight:bold'> SELF-INSTRUCT </span>) 저자들은 LLM 이, 자신이 생성한 generation 을 bootstrapping 하는 기법을 통해 instruction-following 능력을 개선시키는 framework 을 제시한다.
- (**Pipeline**) Pipiline 은 instruction, input, output sample 을 generate 한 뒤, invalid 하거나 이전과 유사한 것들을 filter 하는 형식이다.
- (**Experiment**) 이 방법으로 vanilla GPT-3 에 적용했을 때, private user data 와 human annotation 을 배운 InstructGPT (text-davinci-001) 를 뛰어넘는 성능을 보인다.
- (**Broader Impact**) SELF-INSTRUCT 는 <span style='background-color: #dcffe4'> almost annotation-free method </span> 로 PLM 은 instruction 에 aligning 할 수 있게 하며, 추후 instruction tuning 에 사용될 수 있는 synthetic dataset 을 생성하여 release 하였다.

## Introduction
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/e6d23d09-9ab2-49d9-9d29-3895d5051ca1)


<span style='color:green;font-weight:bold'> Lack of instruction tuning data </span>
<br>
최근 NLP 에서는 LLM 의 강력함을 목격하였다. 
그 중심에는 두 가지 key component 가 있는데, (1)LLM 모델과 (2) human-written instruction data (e.g. PROMPTSOURCE, SUPERNATURALINSTURCTIONS, SUPERNI) 이다.
<span style='background-color: #ffdce0'> 그러나 instruction data 를 collecting 하는 것은 매우 costly 하고, annotator 가 입맛에 맞는 task 는 유명한 task들이기에 limited diversity 를 갖는다. </span>
따라서 instruction tuning process 를 위한 대체 방안이 필요하다.

<span style='color:green;font-weight:bold'> SELF-INSTRUCT </span>
<br>
이 논문에서는 모델 자체의 instructional signal 로 부터 instruction tuning 을 진행하는 semi-automated process 인 <span style='background-color: #dcffe4'> SELF-INSTRUCT </span> 방법을 제안한다. 
위의 그림이 SElF-INSTRUCT 방법론의 overall process 이다.
- (1) 우선 제한된 수의 seed task set (human-written) 으로 시작하여, 새로운 new task 를 위한 instruction 을 생성하게 한다. 이 과정에서 기존의 instruction 들의 collection 으로 부터, new task 를 위한 broad-coverage instruction 을 생성하게 한다.
- (2) 이후 생성된 instruction 을 바탕으로, 모델은 input 과 output 을 생성하게 된다.
- (3) 마지막으로, vlow-quality 와 repeated instruction 을 제거하기 위한 다양한 heuristic 을 통해 filtering 을 진행한다.
- (4) 이 과정은 task 의 수가 원하는 정도로 많아질 때 까지 반복된다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/8ea2833e-91f7-4bb7-8a82-f0a1a6f2d93c)

<span style='color:green;font-weight:bold'> Experiments </span>
<br>
저자들은 SELF-INSTRUCT 방법을 Vanilla GPT-3 에 적용하였다.
이 방법을 통해 52K instruction 과 82K input-output pair 를 생성하였다.
위의 그림과 같이 다양한 범위의 creative 한 task 들을 생성하는 것을 볼 수 있다. (typcial NLP task 과구분되는)
SELF-INSTRUCT 가 적용된 GPT-3 는 SUPERNI 등의 typical NLP 뿐 아니라, 새로운 instruction task 에 대해서도 InstructGPT001 을 이기는 정도의 성능을 보여준다.

## Method
# Defining Instruction Data
Instruction $I_t$ 는 $t$-th task 에 대해, input-output instance $(X_{t,i}, Y_{t,i})$ 를 갖는다.
모델 $M$ 은 $M(I_t, X_{t,i})=Y_{t,i}$ 를 생성한다.
Instruction 과 Input 사이의 boundary 는 엄격하게 두지 않았다. 
예를 들어, "write an essay about school safety" 자체가 instruction 일 수도 있고, "write an essay about the following topic" 이 instruction 이고 "school safety"가 input 으로 주어질 수도 있다.
따라서 Input $X$가 empty 로 주어지는 경우도 나올 수 있다.

# Automatic Instruction Data Generation
첫 번째 Figure 에서 볼 수 있듯이, SELF-INSTRUCT 는 네 가지 pipeline 을 가진다.
- 1) generating task instructions
- 2) determining if the instruction represents a classification task
- 3) instance generation with either an input-first or output-first appraoch
- 4) filtering low-quality data
 
<span style='color:green;font-weight:bold'> (1) Instruction Generation </span>
<br>
우선, 작은 크기의 seed set 으로 시작한다. 저자들은 175개 task 에 대해, 하나의 instruction 과 하나의 instance 로 시작한다.
각 step 마다 8 개의 task instruction 을 in-context example 로 하여 prompt 를 구성한다.
prompt 는 아래와 같다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/ca8b9be4-0ac8-428c-8922-a2ec899e3e6c)

<span style='color:green;font-weight:bold'> (2) Classification Task Identification </span>
<br>
Classification setting 이냐 아니냐가 중요한 요소이므로, 두 번째 step 으로는 생성된 instruction 이 classification task 인지 아닌지를 구분한다.
아래 그림의 prompt 를 이용하여 few-shot ICL 로 모델을 이용한다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/5ae87ac6-2ed5-4c70-a0da-e2c053ca883f)

<span style='color:green;font-weight:bold'> (3) Instance Generation </span>
<br>
주어진 instruction 를 바탕으로 instance 를 생성하는 단계다.
이 과정이 가장 challenging 한데, 그 이유는 <span style='background-color: #dcffe4'> (1) target task 에 대한 이해가 필요하고, (2) input field 에대한 이해와 생성을 해야하며, (3) output 을 완성시켜 생성할 수 있어야하기 때문이다. </span>
즉, 세 가지 단계를 한 번에 해낼 줄 알아야하는 step 이다.

저자들은 우선, instruction-input-output 형태로 주어지는 in-context example 을 통해 LM 이 이 능력을 갖추고 있음을 확인하였다.
이 방법을 저자들은 **INPUT-First Approach** 로 명명하여 사용한다.

그러나 저자들은 이후, 이 방법이 (classification task 에서 특히) 하나의 label 로 bias 된 output 을 생성한다는 것을 발견한다.
이에 저자들은 **OUTPUT-First Approach** 를 제안하는데, possible class label 을 먼저 generate 한 후, 이것과 instruction 을 활용해 input 을 생성하게 하는 것이다. Output-first approach 의 prompt 의 일부는 아래의 그림과 같다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/5cb80608-e2f1-4c43-9ecc-eb61b7a49dd4)

저자들은 <span style='background-color: #dcffe4'> classification setting 일 경우 OUTPUT-First approach 를, non-classification setting 일 경우 INPUT-First Approach 를 적용하였다. </span>

<span style='color:green;font-weight:bold'> (4) Filtering and Postprocessing </span>
<br>
이렇게 생성된 새로운 instruction-instance 를 task pool 에 추가하기 전 filtering 과정을 거친다.
우선, 너무 유사한 task 를 방지하기 위해, ROUGE-L similarity 가 0.7 이상이면 거르도록 하였다.
또 exact same Input, Ouput 이 발견된 경우도 거른다.
물론 heuristic 하게, instruction 이 너무 짧거나 너무 길거나 하는 등의 invalid generation 도 filter out 된다.

# Finetuning the LM to Follow Instructions

이렇게 생성된 diverse large-scale instruction data 를 학습시킨다.
다양한 task 에 대해 robust 하게 만들기 위하여, 다양한 template 의 prompt 를 사용한다.

## SELF-INSTRUCT Data from GPT-3
GPT-3 를 활용하여 SELF-INSTRUCT 를 구현해 생성한 dataset 의 통계이다.

# Statistics

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/f71e67bd-3058-4c7c-8068-a0d70ce5802e)

52K instruction 과 82K 정도의 instance 가 생성되었따.

# Diversity
Berkeley Neural Parser 를 활용해 verb-noun structure 를 구성한 뒤, 분석 한 결과, 아래 그림과 같이 다양한 종류의 instruction 이 생성됨을 볼 수 있다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/06937d6d-1bf7-44b5-8142-9383d364b9a0)

또한 시작 point 의 175개 seed task 와 얼마나 겹치게 생성되었는지를 확인하기 위해, 아래의 ROUGE-L overlap 을 보면, overlap 이 크지 않은 창의적인 task 들이 많이 생성된 것을 볼 수 있다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/c0202ec2-e614-4b95-83da-8bd7e5898af1)

마지막으로, instruction, non-empty input, output 의 길이에 대한 분석은 아래의 그림에서 볼 수 있다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/0d88b501-07b2-4a4f-879c-4761ae798908)

# Quality 

Quality 를 평가하기 위해, 200개의 instruction (각 1개의 instance)를 추출하여 annotator 에게 평가를 시킨 결과, "most of the generated instructions are meaningful" 의 결과를 얻었다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/e3bf9e58-52fd-472a-aa65-aa8a37365916)

## Experiemental Results

SELF-INSTRUCT 로 생성한 data 를 학습한 모델을 $GPT-3_{SELF-INSTRUCT}$ 라고 명하고 아래의 baseline 들과 비교한다.

- Off-the-shelf LMs
T5-LM, GPT-3

 - Publicly available instruction-tuned models
[T0](https://arxiv.org/abs/2110.08207), [T$k$-INSTRUCT](https://aclanthology.org/2020.findings-emnlp.90/) 

- Instruction-tuned GPT3 models
INSTRUCTGPT (text-davinci-001)

# Experiment 1 : Zero-Shot Generalization on SUPERNI Benchmark
Instruction following task 인 SUPERNI benchmark 에 대한 실험결과이다. 이 실험은 대체로 <span style='background-color: #dcffe4'> zero-shot setting  </span> 으로 실험하였다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/80d77626-ea5c-4528-ab04-2f6ffc51d1d9)

- SELF-INSTRUCT 는 GPT-3 의 instruction-following 을 크게 boost 시킬 수 있다.
- InstructGPT001 과 거의 유사한 성능을 보인다.

# Experiment 2 : Generalization to User-oriented Insutrctions on Novel Tasks

Practical usage 에 대한 검증을 위하여, User-oriented Instruction set 을 curate 하여 bnechmark 로 활용한다.
우선, email writing, social media, entertainment, programming 등 LLM 이 자주 쓰일만한 분야를 선정한 후, instruction-instance를 
 craft 한다.
이렇게 252 개의 instruction (각 1개의 instance) 를 생성하였다.
아래 그림에서 small portion 을 볼 수 있다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/aa3e8123-442f-445f-975e-03b36cdc481f)

실험 결과는 아래와 같다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/70dad7d9-b70a-46bf-a6da-55cbe805ad7a)

A,B,C,D 로 나누어 평가를 한 결과 (A rank 일 수록 좋은 평가) text-davinci-001 정도까지는 유사한 결과를 얻을 수 있었다

# Effect of Data Size and Quality

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/984a2805-c56d-4ddd-adb5-961c61579922)

위의 그림의 파란 선에서, data size 가 커질 수록 consistent 하게 성능이 증가하는 것을 볼 수 있다.
그리고, 주황색 점에서, output 을 text-davinci-003 이 생성하게 하여 높은 quality 의 데이터를 생성하게 하였을 때 훨씬 좋은 성능을 얻는다. 이를 통해, 생성되는 data 의 quality 가 좋아질 수록 성능이 좋아질 수 있으므로, 아직 발전의 여지(room)가 이 충분하다고 본다.

##  Conclusion
```
We introduce SELF-INSTRUCT, a method to improve the instruction-following ability of LMs via their own generation of instruction data. On experimenting with vanilla GPT3, we automatically construct a large-scale dataset of 52K instructions for diverse tasks, and finetuning GPT3 on this data leads to a 33% absolute improvement on SUPERNI over the original GPT3. Furthermore, we curate a set of expert-written instructions for novel tasks. Human evaluation on this set shows that tuning GPT3 with SELF-INSTRUCT outperforms using existing public instruction datasets by a large margin and performs closely to InstructGPT001. We hope SELF-INSTRUCT can serve as the first step to align pretrained LMs to follow human instructions, and future work can build on top of this data to improve instruction-following models.
```
