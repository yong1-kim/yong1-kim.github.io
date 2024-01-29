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



<span style='color:green;font-weight:bold'> 초록색볼드체 </span>

<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
