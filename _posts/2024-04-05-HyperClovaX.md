---
layout: post
title: "[Arxiv 2404]HyperCLOVA X Technical Report"
date: 2024-04-05 20:40:00 +0900
use_math: true
categories: [LLM, PLM]
---

[[pdf]](https://arxiv.org/pdf/2404.01954.pdf)  &emsp;
[[hyperclobax]](https://clova.ai/hyperclova)

**NAVER Cloud**
<br> HyperCLOVA X Team  &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/11f4ca10-ce5c-44e5-a67f-909aea3bbf30)

### Abstract
- (**HyperCLOVAX**) 한국어와 한국문화에 학습된 LLM인 **HyperCLOVAX** 를 소개한다. 한국어와 영어, 그리고 코드 데이터셋을 학습하여 특화되어있다.
- (**Evaluation**) Comprehensive reasoning, knowledge, commonsense, factuality, coding,
math, chatting, instruction-following, and harmlessness 등 많은 benchmark 에 대해, 한국어와 영어 모두 실험을 진행하였고, <span style='background-color: #dcffe4'> 한국어에서 매우 강력한 reasoning 능력을 보여준다. </span>
- (**Multilingualism**) 한국어-영어 bilingual 특성 뿐 아니라, Multilingualism 로의 확장으로 기계 번역 등 다양한 언어로의 확장 가능성을 제시한다.

### 1. Introduction
<span style='color:green;font-weight:bold'> ▶ Bias in English Corpus</span>
<br>
현재 다양한 LLM 들이 매우 좋은 성능을 보여주고 있지만, 대부분 North American culture 와 영미권 문화에 강하게 bias 가 되어있다.
이는 pretrianing corpus 가 대부분 영어로 되어있기 때문이다.
<span style='background-color: #ffdce0'> 따라서 한국어와 같은 non-English 언어에 대해서는 특정한 문화나 지리적인 특성 등을 반영하지 못하여 매우 압도적인 성능을 보여주지 못한다. </span>

<span style='color:green;font-weight:bold'> ▶ HyperCLOVA X </span>
<br>
이에 저자들은 HyperCLOVA X family 를 공개한다.
<span style='background-color: #dcffe4'> 이는 강력한 버전인 HCX-L 과 lightweight 버전인 HCX-S 로 구성되어있다. 
 </span>
두 모델 모두 한국어와 한국 문화적인 내용에 맞춰져 있으며(tailored),  영어 외의 다양한 언어에 대하여 좋은 성능을 보인다.
<span style='background-color: #dcffe4'> 모델들은 한국어, 영어, 그리고 코드 데이터셋에 공평하게(evenly) 학습이 되었다.
 </span>

<span style='color:green;font-weight:bold'> ▶ Reasoning Capability </span>
<br>
HyperCLOVA X 모델은 reasoning, knowledge, commonsense, factuality, coding, math, chatting, instruction-following, harmlessness 등 9개의 task 에 대하여 한국어/영어에서 매우 좋은 성능을 보인다.
특히 현존하는 closed-source 와 open-source 를 모두 포함하여, 한국어에 대해서는, 기존 모델들을 뛰어넘는 포괄적인 이해능력을 보여준다.

<span style='color:green;font-weight:bold'> ▶ Multilingual Capability </span>
<br>
또한, 한국에서 자주 사용되는 세가지 다른 언어에 대해 기계번역을 통한 cross-lingual reasoning 능력을 실험하였을 때, state-of-the-art 수준의 machine translation 성능을 보인다.
HyperCLOVA X 의 이러한 인상적인 multilingual ability 는 한국어-영어의 cross-lingual trasnfer 에 대해, <span style='background-color: #dcffe4'> 하나의 언어에 대한 instruction tuning 이 다른 언어에 대하여 intruction-following 능력을 나타내는 emergent ability 를 보인다. </span>

<span style='color:green;font-weight:bold'> ▶ Safety </span>
<br>
Safety 에 대한 보장을 위해, red teaming 기법을 활용하였고, safety data collection process 가 NAVER AI Ethics 원칙에 강하게 기반되었다.
다양한 safety evaluation (automatic & human evaluation) 으로 안정성을 보장한다.

### 2. Training Details
## 2.1. Pretraining

## 2.2. Alignment Learning
# 2.2.1. Supervised Fine-tuning (SFT)

# 2.2.2. Reinforcement Learning from Human Feedback (RLHF)

# 2.2.3. The Alginment Learning Pipeline

### 3. Core Benchmarks
## 3.1. Comprehensive Korean LLM Benchmarks

## 3.2. Comprehensive English LLM Benchmarks

## 3.3. Commonsense Reasoning

## 3.4. World Knowledge and Factuality

## 3.5. Mathematics

## 3.6. Coding Capabilities

## 3.7. Chat and Instruction-Following

## 3.8. Harmlessness

## 3.9. Comparison with Closed Source Models

### 4. Multilinguality
## 4.1. Cross-Lingual Reasoning

## 4.2. Machine Translation

## 4.3. Cross-lingual Transfer

### 5. Safe and Responsible AI

## 5.1. HyperCLOVA X Ethics Principles

## 5.2. Red Teaming and Safety Data Collection

## 5.3. Safety Evaluation

# 5.3.1. Toxicity

# 5.3.2. Social Bias

# 5.3.3. Human Evaluation

### Conclusion
```
HyperCLOVA X represents a significant advancement in LLMs, particularly emphasizing the Korean language and culture while maintaining strong capabilities in English and other languages. Through a training process that incorporated a balanced mix of Korean, English, and programming languages, followed by supervised fine-tuning and reinforcement learning from human feedback, HyperCLOVA X demonstrates exceptional proficiency in a variety of tasks.

HyperCLOVA X’s performance across a wide range of benchmarks—e.g. reasoning in Korean and English, and problem-solving in coding and math—showcases its capacity and versatility. Also, its impressive multilingual ability, especially in cross-lingual reasoning and machine translation, further illustrates its generalization capability and the potential for broad application across different linguistic contexts.

Moreover, the commitment to responsible AI development and deployment is manifested through the extensive safety evaluations and adherence to ethical principles. HyperCLOVA X’s sophisticated handling of toxicity, social biases, and other ethical concerns through systematic red teaming and safety data collection processes, along with its performance in human evaluation studies, highlight its potential as a safe and reliable AI assistant. Overall, HyperCLOVA X sets a new standard for bilingual and multilingual LLMs, paving the way for more inclusive and culturally sensitive AI technologies.

As future work, we intend to explore multimodality, aiming to broaden HyperCLOVA X’s capabilities to seamlessly process and integrate diverse types of data, such as text, images, and audio. Moreover, we are set to explore the efficacy of model quantization techniques, with the goal of optimizing HyperCLOVA X ’s inference without sacrificing its accuracy or the quality of the output. Additionally, we are actively researching the integration of external tools and APIs to augment the model’s functionalities. This will enable HyperCLOVA X to access specialized datasets and services, significantly enriching and enhancing the factuality of its responses. Our team is committed to integrating these innovative research topics with the existing and future services at NAVER and its subsidiaries as we strive to advance AI technologies that benefit humanity
```

<span style='color:green;font-weight:bold'> 초록색볼드체 </span>
<br>
<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>

<span style='color:green;font-weight:bold'> ▶ </span>
<br>
