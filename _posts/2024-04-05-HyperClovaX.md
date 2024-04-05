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
HCX-L 과 HCX-S 모두 한국어/영어/코드 데이터셋에 pretraining 된 이후, Supervised Fine-tuning (SFT) 와 reinforcement learning form human feedback (RLHF) 를 통해 instruction-following ability 가 향상되었다.

## 2.1. Pretraining
HYPERCLOVA X 는 [HYPERCLOVA](https://aclanthology.org/2021.emnlp-main.274/) 의 updated version 이며, trasnformer decoder 에 약간의 modification 이 추가된 버전이다.
Context Length 향상을 위해 position embedding 으로 [rotary position embeddings](https://arxiv.org/pdf/2104.09864.pdf) 을 활용하였고, pre-normalization 과 [grouped-query attention](https://aclanthology.org/2023.emnlp-main.298.pdf) 을 사용하였다.


<span style='color:green;font-weight:bold'> Data </span>
<br>
Pretraining data 는 한국어(Korean), Multilingual, Code segment 로 이뤄져 있다.
Multilingual 은 대부분 영어로 이뤄져있지만, 일본어, 독일어, 프랑스어 등 다양한 언어로도 이뤄져 있고, 한국어에 특화시키기 위하여, 한국어 데이터셋을 전체 데이터 크기의 3 분의 1 이 되게 확보하였다.
<span style='background-color: #dcffe4'> 
결과적으로, 한국어, multilingual, code 데이터 세 개가 equal distribution 을 갖는다.
 </span>
데이터 퀄리티를 위하여 반복적인 문장, 너무 짧은 문장, 너무 낮은 퀄리티의 document 는 제외하였고, Personallyh identifiable information (PII); 개인 정보등은 제거하였다.
<span style='background-color: #dcffe4'> 
또한, Knowledge-containig data 를 upsample 하여 performance 향상을 이끌어낸다. </span>

 <span style='color:green;font-weight:bold'> Tokenizer </span>
<br>
한국어 중심의 LLM을 위해 효과적인 Tokenizer 준비하는 것이 중요하다. 한국어는 어근 의미 형태소에 문법 형태소를 붙여 단어를 형성하는 응집형 언어이다. HyperCLOVA X 는 형태소 인식 byte-level BPE를 훈련하여 한국어 문서를 효율적으로 토큰화한다.
아래 표에서 한국어에 강력하게 효율적임을 볼 수 있다.
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/312d4e30-98b6-4f02-851e-0d46d57663ee)

<span style='color:green;font-weight:bold'> Pretraining Scheme </span>
<br>
Left-to-Right 에 한정짓지 않고, [PSM & SPM](https://arxiv.org/pdf/2207.14255.pdf) training 을 활용한다. (fill-in-the-middle 방법이다)
이 학습 방법은 pre-training 동안 in-filling performance 를 얻기 위해서 고안된 것이다.
90% 학습은 4096 context length 로 학습하고, 나머지 10% 는 32768 length 로 학습한다.
또한 [flash attention](https://arxiv.org/pdf/2205.14135.pdf) 과 3D parallelism 을 활용하며, bf16 precision 을 활용한다.

## 2.2. Alignment Learning
# 2.2.1. Supervised Fine-tuning (SFT)
각각의 prompt 에 대하여 completion 의 likelihood 를 maximize 하게 SFT 를 통한 alignment learning 을 한다.
이를 통해 instruction-following, problem-solving, coding, creative writing 능력 등을 향상시킨다.

SFT 데이터셋에는 ‘<|user|>’, ‘<|assistant|>’, ‘<|endofturn|>’ 세 가지 special token 을 추가하여 turn 을 구분한다.
<span style='background-color: #dcffe4'> 
Multiturn sample 학습을 위해서는 assistant turn 을 제외한 나머지 turn 들에는 loss masking 을 적용한다. </span>
SFT 학습에서 효율적인 GPU 활용을 위해, 효율적인 batching strategy 를 구사한다.

# 2.2.2. Reinforcement Learning from Human Feedback (RLHF)
SFT 만을 이용한 Alignment tuning 이 uninformative 하거나 harmful content 를 포함하는 것은 이제 공공연한 사실이다.
이를 위해 대부분 RLHF 는 3H value 인 helpful, honest, harmless 를 학습시킨다.
HyperCLOVA X 는 Proximal Plicy Optimization (PPO) 를 활용하였다.

<span style='color:green;font-weight:bold'> Reward Model. </span>
<br>
SFT 학습이 끝난 모델에, random 하게 init 된 linear head 를 붙여 scalar reward 를 내뱉게 한다.
모델은 Bradley-Terry model 에 기반한 ranking loss 로 학습되는데, 이는 chosen 과 rejected 의 차이를 reward negative log-likelihood 를 최소화하는 방법이다.
이 모델은 한 에폭만 학습된다. ([InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) 논문에 기반)

<span style='color:green;font-weight:bold'> Reinforcement Learning </span>
<br>
<span style='background-color: #dcffe4'> 
다른 모델들과 유사하게 PPO 를 활용하였고, KL penalty term([[1]](https://arxiv.org/abs/1907.00456),[[2]](https://proceedings.neurips.cc/paper_files/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf))을 0.04 계수 와 함께 reward 에 추가한다. </span>
Policy Network 는 post-SFT model 이고, reward model 은 앞서 언급한 모델이다.


<span style='background-color: #ffdce0'> 
많은 기존 연구([AlpacaFarm](https://arxiv.org/pdf/2305.14387.pdf), [[3]](https://arxiv.org/pdf/2310.03716.pdf), [[4]](https://arxiv.org/pdf/2307.04964.pdf))에서 RLHF 이후 output length 의 증가를 report 하였다.  </span>
저자들 또한 같은 현상을 목격하였고, model 이 longer sequence 를 좋아하는 경향을 알아낸다.
이를 해결하기 위해 <span style='background-color: #dcffe4'> iterative human feedback </span> 방법을 고안한다.
또한, 특정한 length 와 format 에 한정된 instruction set 에 overfitting 되지 않기 위해, early stopping mechanism 을 추가하였다.

<span style='background-color: #ffdce0'> 
또한, Transformer 기반의 LLM 은 repetition 에 취약하다.  </span>

저자들은 역시 이 문제도 발견하였고, <span style='background-color: #dcffe4'> PPO 에 sequence-level unliklihood training 를 추가하여, 최소한의 추가적인 training cost 로 repeition 문제를 해결하였다. </span>

PPO 의 경우, 통상적으로 SFT 보다 네 배의 시간을 요구한다.
이 과정을 optimize 하게 위하여, multi-node setting 으로 asynchrnous processing 을 통해 process 를 병렬화한다.
<span style='background-color: #dcffe4'> 
특히 각 iteration 의 rollout phase 에서 네 개의 네트워크에 inference 를 하기 위한 continous batching 을 employ 한다.
 </span>

# 2.2.3. The Alignment Learning Pipeline
특정 checkpoint 에서 model 의 training 을 interuppt 하는 대신, check-point saving event 를 발견하고, 다른 computation resource 에서 asynchrnous 하게 evaluate 하는 **event-driven pipeline** 을 통해 효율적인 학습을 진행한다.

또한, SFT, RM, PPO learning process 를 하나의 스텝 이후에 자동적으로 시작되게하여 human intervention 을 최대한 줄인다.

### 3. Core Benchmarks

<span style='color:green;font-weight:bold'> Benchmark Design. </span>
<br>

Multilingual  언어 모델의 발전에서 큰 constraint 는 영어 이외의 언어에 대한 철저한 평가 프레임워크의 부재이다. 특정 언어의 능력은 linguistic proficiency 뿐만 아니라 해당 언어 사용자에게 독특한 문화적 및 사회적 뉘앙스에 대한 깊은 이해도 필요하다. HyperCLOVA X 의 언어 능력을 평가하기 위해, 내/외부적으로 찾은 영어와 한국어 벤치마크를 활용한다.

Reasoning, world knowledge, and mathematics transcend language 과 같은 핵심 역량은 언어를 초월하기 때문에(언어에 특화되지 않아도 되므로), 이런 벤치마크의 상당 부분은 언어 중립적 기술을 평가하기 위해 영어로 진행된다. 한편, 언어별 질문에 대한 다양한 측면을 모델이 얼마나 잘 포함하는지와 문화적 뉘앙스를 다루는 모델의 능력을 평가하기 위해, 각 언어에 맞게 구성된 두 가지 상세한 벤치마크 카테고리를 활용한다.

또한, 한국어 데이터셋은 기계 번역된 것을 활용하지 않고, 전문가에 의해 세심하게 제작된 것을 활용하거나 이미 그렇다고 인정받은 것들을 활용한다. 이러한 벤치마크에는 KoBigBench (KBB)와 같은 지역 특화 질문과 내부 노력에서 구축된 포괄적인 한국어 벤치마크인 KMMLU 내의 한국어 특정 질문 세트가 포함되어 있어 모델의 한국 문화 및 사회적 맥락 이해를 엄격하게 평가한다. 

<span style='color:green;font-weight:bold'> Baselines. </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/d248a5a8-5b6c-4961-bc9d-1fdea627d7a6)

HyperClOVA X 는 한국어와 영어 모두에 내재적 효율성을 위해 학습되었기 때문에, 그 평가 역시 counterpart 와의 직접적인 비교가 어렵다.
따라서, 한국어 유창성에 관련한 비교는 한국어특화 LLM 들과 비교하고, langauge-agnostic task 에 대해서는 일반적인 foundational model 들과 비교한다. 한국어 평가를 위해, Korean LLM community 에 만연한 비교 방법인 Korean corpus 로 학습 된 후 target language 에 적용하는 방법으로 closed-, open- source LLM 들과 비교한다.

- **Models Specializing in Korean** : (1) [Polyglot-Ko(TUNiB)](https://arxiv.org/pdf/2306.02254.pdf), (2) [SOLAR|SOLAR-chat(Upstage)](https://arxiv.org/pdf/2312.15166.pdf) (LLaMa2 아키텍쳐에 Mistral parameter 로 init), (3) LLaMa2 Ko|LLaMa2 KoEn(huggingface), (4) [KORani(Krafton-ai)](https://github.com/krafton-ai/KORani), (5) [EEVE-Korean-v(yanolja)](https://arxiv.org/pdf/2402.14714.pdf) (SOLAR 에 한국어를 위한 효율적인 vocab 활용한 모델)

- **General Foundation Models** : (1) Falcon, (2) LLaMA2, (3) Mistral 7b

<span style='color:green;font-weight:bold'> Evaluation Methods. </span>
<br>
두 가지 main evaluation method 를 택한다.
- (1) Open-ended question-answering | free-form answer ( BigBench-Hard )

※ 자세한 세팅은 논문 참고

- (2) Closed-ended question-answering | candidate answer




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
