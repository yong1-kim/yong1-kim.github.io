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





<span style='color:green;font-weight:bold'> 초록색볼드체 </span>

<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
