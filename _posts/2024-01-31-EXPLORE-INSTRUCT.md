---
layout: post
title:  "[EMNLP2023] EXPLORE-INSTRUCT: Enhancing Domain-Specific Instruction Coverage through Active Exploration"
date:   2024-01-31 17:47:00 +0900
use_math: true
categories: [LLM, PLM]
---

[[pdf]](https://aclanthology.org/2023.emnlp-main.587.pdf) &emsp;
[[github]](https://github.com/fanqiwan/Explore-Instruct)

**Fanqi Wan<sup>1*</sup>, Xinting Huang<sup>2†</sup>, Tao Yang<sup>1</sup>, Xiaojun Quan<sup>1†</sup>, Wei Bi<sup>2</sup>, Shuming Shi<sup>2</sup>**
<br><sup>1</sup> School of Computer Science and Engineering, Sun Yat-sen University, China <sup>2</sup> Tencent AI Lab &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/2aa06fa7-6a45-4214-8ce2-bc4db30177f7)

## Abstract
- (**Diversified Instruction Tuning**) Instruction Tuning 은 broader specturm 의 task 로 diversity 를 학습할 수 있다.
- (<span style='background-color: #ffdce0'> Lack of Data </span>) 그러나, 현존하는 data 로는 individual domain 에 대한 data 까지의 coverage 는 불가능하다.
- (<span style='color:green;font-weight:bold'> EXPLORE-INSTRUCT </span>) 이 논문에서는 LLM 의 exploration 을 통해, domain-specific instruction-tuning 를 위한 data coverage 를 발전시키는 방법론을 제안한다. 이 방법론은 몇몇 대표적인 domain 으로부터, diversified and domain-focused instruction-tuning data 를 생성할 수 있다.
- (**Experiment**) 생성한 data 에 대한 평가에서, 넓은 diversity 를 커버할 수 있는 것을 확인할 수 있으며, baseline 에 적용되었을 때, domain sepcific data enahncement 를 보였다.

## 1. Introduction
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/aef0547c-ca1d-4cdb-a5cb-a137b18f154d)

<span style='color:green;font-weight:bold'> ▶ 기존의 다양하지 않은 instruction-tuning data </span>
<br>
Large Language Model(LLM) 은 instructino 을 통해 다양한 영역의 문제를 해결할 수 있음을 보인다.
따라서 중요한 challenge 는 diverse instruction-tuning data 를 construct 하는 것이다.
기존에 많이 사용되는 human-curated instruction-tuning data 로는 [FLAN](https://arxiv.org/abs/2210.11416), [SUEPR-NATURALINSTURCTION](https://aclanthology.org/2022.emnlp-main.340/) 등이 있다.
최근 [SELF-INSTRCUT](https://arxiv.org/abs/2212.10560) 논문에서 instruction-tuning data 의 diversity 를 amplify 하는 방법론을 제안하기도 하였다.

이러한 general 한 instruction-tuning data 에 대비하여, domain-specific instruction-tuning data 를 자연스럽게 정의할 수 있다.
Human-curated instruction-tuning data 나 Machine-generated data  모두 활발히 연구가 되고 있지만, 이러한 것들이 wide range 를 커버하지는 못하고 있는 실정이다.
이러한 이유는 대부분이 human curation 에 over-reliance 하고 있으며, popular NLP task 에 bias 되어 있기 떄문이다.
위의 Figure 에서 보듯이, Human-curated (핑크색) 과 SELF-INSTRUCT (하늘색) 는 다양한 범위를 커버하지는 못하는 것을 볼 수 있다.

<span style='color:green;font-weight:bold'> ▶ EXPLORE-INSTRUCT </span>
<br>
이러한 instruction-tuning data creation 에서의 다양성 확보를 위해 EXPLORE-INSTRUCT 방법론을 제안한다.
저자들은 우선 <span style='background-color: #dcffe4'> domain space 는 내재적으로 tree 의 구조를 가지고 있다</span>고 판단한다.
Classical Search Algorithm 과 LLM 의 강력한 파워의 결합으로, EXPLORE-INSTRUCT 는 domain space 를 traverse 하며 instruction-tuning data 를 생성한다.

이 방법론은 두 가지 _(1)lookahead (2)backtracking_strategy 를 취한다.
첫 번째는 fine-grained sub-task 를 철저히 조사하며, backtraking 은 domain specturm 을 넓히기 위하여 alternative branch 를 찾는다.
저자들은 Depth-first search(DFS)의 간단한 방법으로 넓은 domain space 를 찾아낼 수 있음을 보인다.

<span style='color:green;font-weight:bold'> ▶ Validation of EXPLORE-INSTRUCT </span>
<br>
이들은 EXPLORE-INSTRUCT 방법론의 평가를 위하여, *rewriting, brainstroming, math* 의 세 가지 이질적인 domain 을 선택하여 testbed 로 활용한다.
이들은 각각 unique use case 와 different skill set 을 필요로 한다.
EXPLORE-INSTRUCT 를 적용하였을 때, 각가의 domain 에서 엄청난 수준의 넓은 coverage 를 갖는 것을 확인한다.
<span style='background-color: #dcffe4'> 이 과정에서 LLM 이 넓은 범위의 domain 을 탐색할 수 있을 뿐 아니라, 각각의 task 를 깊게 이해하여 fine-grained sub-task 로 decompose 할 수 있음을 보인다. </span>
이렇게 생성된 instruction-tuning data 를 fine-tuning 하였을 때, baseline 을 뛰어넘는 성능을 보인 것을 확인한다.

## 2. Method
# 2.1. Domain Space Representation
저자들은 domain-specific instruction 의 coverage 에 대한 개념을 다음 두 가지로 정리한다.
- **breadth** : domain 속의 different task category 를 의미
- **depth** : fine-grained task decomposition 을 의미

 Breadth 를 이해한다는 것은 domain 속의 여러 catgeory 를 이해하는 능력을 가진 것이고, depth 를 이해한 다는 것은 task decomposition 을 통해 precise problem solving 을 할 수 있다는 것이다.

따라서 저자들은 tree structrue $T$ 는 task nodes $V$ 와 sub-task 와의 edge $E$ 로 이뤄진다고 가정한다.


# 2.2. Active Exploration Strategy

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/950ca60f-caaa-4457-bf7d-bcbb8e5653d9)

실제 EXPLORE-INSTRUCT 알고리즘은 두 가지 (1)lookahead, (2)backtracking exploration 으로 이뤄진다.

<span style='color:green;font-weight:bold'> (1) Lookahead Exploration </span>
<br>
Lookahead는 depth 를 파고드는 exploration 으로 다시 말해, fine-grained sub-task 를 mapping out 하는 것이다.
즉 task $V$ 에 대해서, lookahead exploration 은 LLM 을 활용하여 M 개의 sub-task 를 만드는 과정이다.
lookahead prompt 는 아래와 같다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/866bdb36-1483-4612-bcce-c07d5aa6d953)

<span style='color:green;font-weight:bold'> (2) Backtracking Exploration  </span>
<br>
Backtracking 은 breadth 를 넓히기 위한 것이다.
따라서, given task node $V$ 에 대해서 backtracking 은 부모 노드 $PV$를 찾은 뒤, 그 것에서 LLM 을 활용하여 M 개의 새로운 sub-task 를 찾아내는 것이다.
prompt 는 위의 lookahead 와 비슷하다.

# 2.3. EXPLORE-INSTRUCT Implementation
EXPLORE-INSTRUCT 는 두 개의 process 로 이뤄진다.
<span style='color:green;font-weight:bold'> (1) Domain Exploration Strategy </span>
<br>
위의 lookahead 와 backtracking 을 통해 하나의 root task 의 breadth 와 depth 를 확장하는 것이다.
Stopping crieteria breadth B 와 depth K 를 만족할 때 까지 반복하여 tree 를 확장시킨다.

<span style='color:green;font-weight:bold'> (2) Instruction-Tuning data generation </span>
<br>
이렇게 확장된 domain space tree 를 LLM 을 활용하여 N 개의 instruction 과 corresponding reponse 를 생성한다.
instruction 의 다양성을 확보하기  위하여, diversity filter 도 적용하였다. 
SELF-INSTRUCT 에서 활용한 ROUGE-L overlap 을 filter 로 활용한다.

# 3. Data-Centric Analaysis 
우선 EXPLORE-INSTRUCT 의 효용성을 검증하기 위해 생성되는 data 를 분석해본다.
Baseline 으로는 **(1) Domain-Specific Human_curated** : SUPER-NATURALINSTURCTION , **(2) Domain-Aware Self-Instruct** : Depth K=0 으로 한 EXPLORE-INSTRUCT 방법이다.

# 3.1 Data Statistics
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/20d91e94-e339-4987-8292-05f76d0f2a9d)

위의 표에서 Human-Curation, Domain-Aware Self-Instruct, EXPLORE-INSTRUCT 에 대한 statistics 를 볼 수 있다. 
EXPLORE-INSTRUCT 의 verb-noun pair 의 수가 더 많으면서도, 표준편차는 작은 것을 볼 수 있다.
이러한 현상은 rewriting 과 math domain 에서 두드러지는 반면, brainstroming 에서는 그렇지 못하다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/7877f54c-fb55-42cb-9530-2f1400433931)

비쥬얼라이제이션을 위하여 10 개의 root task 를 정하여 generated instruction 을 비교했을 때, 아래의 EXPLORE-INSTRUCT 가 훨씬 더 다양하게 생성한 것을 볼 수 있다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/9b582aff-9b24-49a9-b830-19c956f6938e)

또한, SELF-INSTRUCT 부터 활용 중인 ROUGE-L overlap 에 관한 figure 는 위에서 볼 수 있다.
역시 핑크색으로 표현된 EXPLORE-INSTRUCt 가 더 넓은 다양성을 보인다.

## 4. Experiment
# 4.1. Benchmarks
EXPLORE-INSTRUCT 가 생성한 세 개의 domain 을 testbed 로 활용한다
- **rewriting** : BELLE test set 으로 부터 생성한 testbed
- **brainstroming** : BELLE test set 으로 부터 생성한 testbed
- **math** : MATH test set 으로 부터 생성한 testbed

# 4.2. Expore-LM and Baseline Models
<span style='color:green;font-weight:bold'> Explore-LM </span>
<br>
EXPLORE-INSTRUCT 가 생성한 instruction-tuning data 를 fine-tuning 한 모델이다. : Ours model
Explore-LM-Ext 는 sampled instance 를 확장하여 fine-tuning 한 extension 모델이다.

<span style='color:green;font-weight:bold'> Baseline Models </span>
<br>
- **Domain-Curated-LM** : 앞서 언급한 Human-curated data 를 학습한 모델
- **Domain-Insturct-LM** : 앞서 언급한 Domain-aware self-instruct data 를 학습한 모델
- ChatGPT

위의 언급되는 모델은 ChatGPT 를 제외하고 모두 LLaMA 7B 를 backbone 으로 하여 fine-tuning 한다.

# 4.3. Resluts and Anlaysis
<span style='color:green;font-weight:bold'> (1) Automatic evaluation results in the brainstorming and rewriting domains. </span>
<br>
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/98c0b965-dfd5-4ffa-945d-e57c0699c40a)

<span style='background-color: #dcffe4'> brainstorming and rewriting 의 domain 에서는 ChatGPT 를 제외한 모델들에 대해 압도적인 성능을 자랑한다. </span>

<span style='color:green;font-weight:bold'> (2) Automatic evaluation results in the math domains. </span>
<br>
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/6c4fbf1a-a835-4b95-a3e1-81e4ee907885)

<span style='background-color: #dcffe4'> ChatGPT 에는 크게 미치지 못하지만, 작은 차이로 baseline model 대비 성능 향상을 이룬다. </span>

<span style='color:green;font-weight:bold'> (3) Human Evaluation. </span>
<br>
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/74e2e6ce-be6f-4f3e-acdb-6ea17b19c38a)

<span style='background-color: #dcffe4'> ChatGPT 에는 지지만 다른 baseline model 대비 우세를 보인다. </span>

<span style='color:green;font-weight:bold'> (4) Data Structure Analysis and Quantity Analsysis </span>
<br>
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/d99279f0-3964-42e5-b8dd-b0d6d4e8a3eb)

<span style='color:green;font-weight:bold'> (5) Data Quantity Analysis </span>
<br>
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/edee1082-e108-4d35-9e5b-f4024efe11bb)

## Conclusion
```
In this work, we introduce EXPLORE-INSTRUCT, a novel approach to enhancing domain-specific instruction coverage. Drawing inspiration from classical search algorithms, EXPLORE-INSTRUCT leverages the power of LLMs to actively explore the domain space and obtain diverse and domain-focused instruction-tuning data. Our experimental results demonstrate the efficacy of EXPLORE-INSTRUCT through data-centric analyses and model performance evaluations in the rewriting, brainstorming, and math domains, highlighting significant enhancements in instruction coverage and superior model performance compared to multiple baseline methods as demonstrated by both automatic and human evaluations.
```
