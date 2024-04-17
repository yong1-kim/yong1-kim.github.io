---
layout: post
title: "[EMNLP2023] Enhancing Chat Language Models by Scaling High-quality Instructional Conversations"
date: 2024-04-17 13:20:00 +0900
use_math: true
categories: [LLM, PLM]
---

[[pdf]](https://aclanthology.org/2023.emnlp-main.183.pdf) &emsp;
[[github]](https://github.com/thunlp/UltraChat)

**Ning Ding<sup>1∗</sup>, Yulin Chen<sup>2,3∗</sup>, Bokai Xu<sup>4</sup>, Yujia Qin<sup>2,3</sup>, Shengding Hu<sup>2,3</sup>, Zhiyuan Liu<sup>2,3†</sup>, Maosong Sun<sup>2,3†</sup>, Bowen Zhou<sup>1†</sup>**
<br> <sup>1</sup> Department of Electronic Engineering, Tsinghua University, <sup>2</sup> Department of Computer Science and Technology, Tsinghua University, <sup>3</sup> BNRIST, IAI, Tsinghua University, <sup>4</sup> The Chinese University of Hong Kong, Shenzhen &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/02a7fabe-3a4c-4303-8fdb-17b6ee11d6a3)

## Abstract
- (**Finetuning chat Language model**) ChatGPT 와 같은 chat language model 을 instruction data 를 통한 fine-tuning 하는 것은, diversity 와 quality 가 받춰줄 때 좋은 성능을 끌어올릴 수 있는 방법이다.
- (<span style='color:green;font-weight:bold'> UltraChat </span>) Human query 를 포함하지 않은 large-scale instructional conversation 을 담고 있는 UltraChat 을 제안한다. 이 데이터셋은 scale, average length, diversity, coherence 등에서 우수성을 보인다.
- (**UltraLM**) UltraChat 을 활용하여 LLaMA 를 finetuning 하여, UltraLM 을 만들었고, WizardLM 과 Vicuna 를 포함한 open-source model 보다 좋은 성능을 보인다.

## 1. Introduction

<span style='color:green;font-weight:bold'> ▶ Chat LLM </span>
<br>
Large Language Model (LLM) 은 놀라울 만한 성능을 보이며, conversation 에 특화된 ChatGPT 와 같은 Chat LLM 은 선풍적인 인기를 끌고 있다.
현재 많은 open-source Chat LM 모델들이 공개되었지만, <span style='background-color: #ffdce0'> ChatGPT 나 GPT-4 는 고사하고 Vicuna 를 이기는 모델 조차 없다. (2023년 5월 기준) </span>

<span style='color:green;font-weight:bold'> ▶ UltraChat </span>
<br>
이 연구에서는 가장 단순한 방법으로 성능을 끌어올릴 수 있다고 믿는다: <span style='background-color: #dcffe4'> Quality and diversity of training data play a vital role in further imporving the performance of chat language model. </span>
다시 말해, 높은 quality 와 더 다양한 data 가 더 좋은 결과를 이끌어낼 수 있다는 것이다.
저자들은 million-scale multi-turn instructional conversation data 인 UltraChat 을 공개한다.
QA 나 Summarization 같은 task 를 활용하여 conversation 을 구성하지 않고, <span style='background-color: #dcffe4'>  Questions about the World, Creation and Writing, Assistance on Existing Material 이라는 세 sector를 curate 한다. </span>
이후 realistic multi-run conversation 을 구성하기 위하여, 두 독립적인 ChatGPT Turbo API 에게 대화를 진행시켜 query 와 response 를 생성하게 시킨다.

<span style='color:green;font-weight:bold'> ▶ Experiment </span>
<br>
LLaMA-13B 모델에 UltraChat 을 학습시켜 UltraLM 을 만들었다.
UltraLM 은 GPT-4 로 평가되었을 때 가장 높은 점수를 기록하였으며, 모든 open-source model 을 능가하는 퍼포먼스를 보인다.

## 2. Related Work
<span style='color:green;font-weight:bold'> Instruction Tuning </span>
<br>
[FLAN-T5](https://arxiv.org/pdf/2109.01652.pdf) 가 60 개의 NLP dataset 을 학습하여, LM 이 instruction tuning 을 통해 instruction following 능력을 갖출 수 있음을 보인 뒤, 많은 모델들이 instruction tuning 을 통해 학습되었다.
[T0](https://arxiv.org/pdf/2110.08207.pdf) 와 [InstructGPT](https://arxiv.org/pdf/2203.02155.pdf) 가 대표적인 예시이고, [FLAN2022](https://arxiv.org/pdf/2301.13688.pdf) 에서는 다양한 task 를 배우는 것이 out-of-distribution 일반화 성능이 좋음을 보였다.
InstructGPT 이후에는 강화학습을 이용한 human preference 학습에 대해서도 많은 연구가 있어왔다.

<span style='color:green;font-weight:bold'> Data Augmentation with LLMs </span>
<br>
Large-scale human-annotated instruction 을 모으는 것은 매우 어려운 일이다.
이를 위해 ChatGPT 나 GPT-3.5 와 같은 well-tuned LLM 으로 부터 sampling 한 data 를 모으는 것이 주목받고 있다.
예를 들어, Self-Instruct 나 Alpaca 는 Text-Davinci-003 을 distilling 하여 high-quality 의 instruction-reponse pair 를 생성한다.
Alpaca 의 성공은 LLM 에 data augmentation 을 부추겼다. 
그 성공으로, code-alpaca, alpacacot, GPT4ALL, ShareGPT, Dolly-v2, BELLE, Vicuna, Koala, Baize 등이 탄생하였다.
CAMEL 의 경우, multi-agent role-play 환경을 통해 real human conversation 을 simulate 한다.

## 3. Data Construction

아래 표는 ChatGPT 를 활용한 direct multi-turn dialog generation 결과와 UltraChat 의 비교이다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/09ab3913-6fca-4a39-9225-db108a4d5650)

대화 데이터의 퀄리티를 결정하는 두 개의 key point 를 발견할 수 있다.
(1) _An opening line determines the topic of the dialogue_
(2) _A user determines the plot of the dialogue, and the output should be tailored to the current topic with diverse language styles and requests_

따라서 기존의 방식대로 comprehensive open-domain instructional chat datset 을 모으는 것과 다르게, <span style='background-color: #dcffe4'> data collection schema 는 interaction 을 잘 capture 할 수 있어야 data quality 를 증가시킬 수 있다. </span>
UltraChat 은 다음의 세 가지 스키마로 design 된 conversation data 를 cover 한다: **(1) Questions about the World, (2) Creation and Writing, (3) Assistance on Existing Materials**.
<span style='background-color: #dcffe4'> Diversity 는 opening line 에 크게 의존하므로, 다양한 set 의 opening line 과 user 를 prompt 하는 방법에 치중되어있는 방법론을 제안한다. </span>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/3ebbe9b4-c775-46e6-9669-c8b82727f044)


# 3.1. Questions about the World
우선 Real world 에 존재하는 concept, object, entity 에 관한 정보를 query 하는데 집중한다.
우선 아래의 Table 2 처럼 ChatGPT 로 하여금 30개의 concept 을 추천받는다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/485a14d5-a1df-4524-bc08-80b469caef78)

이후 30개에서 50개로 subtopic 으로 dive 한다.
마지막으로, 각각 subtopic 혹은 concept 마다 10 개의 다른 질문을 생성하고, ChatGPT 로 하여금 각각 question 을 기반으로 10개의 question 을 더 만들게 한다.

다른 concept 을 결정하는 방법은 Wikidata entity 를 이용하는 방법이다.
가장 많이 등장하는 10,000 개의 entity 에 대하여 5 개의 meta-question 을 생성하고, 각각 10개의 specific question 과 20 extended question 을 생성한다.
이후 filtering 과정을 통해, 500,000 (500K) 개의 질문을 opening line 으로 만든다.

# 3.2. Creation and Writing

두 번째는 email 작성이나 수필/연극 작성 처럼 human-input condition 에 대한 새로운 정보를 생성하는 과정이다.
이는 AI assistant 의 창의성을 활용하는 과정이다.

우선 아래 Table 3 와 같이 20 개의 text material type 을 고른다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/75272132-c994-4785-a0c5-fd98e1b19481)

이후 ChatGPT 를 활용하여 다양한 instruction 을 생성한 뒤, 다시 ChatGPT 를 통해 refine 한다.
이 instruction 은 dialog generation 의 opening line 으로 활용된다.

# 3.3. Assistance on Existing Materials

C4 corpus 에서 text material을 수집하고 다양한 콘텐츠 유형을 위해 수동으로 키워드를 선별하며 텍스트를 URL과 키워드를 일치시켜 분류한다. 그리고 ChatGPT에게 100,000개의 수집된 text material 각각에 대해 다섯 가지 instruction 를 생성하도록 요청하여 opening line으로 총 500,000개의 조각이 생성된다.

# 3.4. User Simulation and Refinement

Dialog History 만을 user model 에게 주면 마치 AI assistant 처럼 대답을 하는 현상이 있다.
이것은 multi-turn conversation 을 만드는데 매우 안좋은 요소가 된다.
따라서 저자들은 user personality 를 추가적으로 부여한다. 
이렇게 Dialog data 가 생성이 된 이후에 filtering 과정을 거친다.

## 4. Data Analysis

# 4.1. Statistical Analysis

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/4ec87d79-6552-413e-9d15-c9b345153bb9)

# 4.2. Human Assessment

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/be800707-31e3-4fed-acfc-a0dbb622bf6a)

※ 자세한 setting 은 논문참조.

## 5. Experiments

LLaMA-13B 모델에 UltraChat 을 학습시킨다.
단순히 dialog 를 적은 sequence 로 쪼개 2048 토큰 안에 들어오게 한 뒤, 일반적인 LM loss 로 학습시킨다.
128 A100 GPU 를 활용하여 512 batch size 로 학습시킨다.

# 5.1. Experimental Setup

<span style='color:green;font-weight:bold'> Baselines </span>
<br>
- Backbone : LLaMA, Pythia
- Baseline : Alpaca, Vicuna, Koala, Dolly, OpenAssistant, WizardLM | ChatGPT, MPT, Biaze


<span style='color:green;font-weight:bold'> Datasets </span>
<br>
- Benchmark Evaluation : ARC-CHallenge, HellaSwag, MMLU, TruthfulQA
- Response Quality Evaluation : GPT-4, AlpacaEval, Evol-Instruct-test

# 5.2. Benchmark Evaluation

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/81abfe18-01d3-43ac-844c-ca4ec671dd47)

- UltraLM는 순수한 지시 튜닝을 통해 UltraChat 데이터셋에서 LLaMA-13B보다 큰 성능 향상을 보이며, 네 가지 벤치마크에서 SOTA 를 보인다. 이는 UltraLM이 World knowledge 와 commonsense knowledge 에 대한 광범위하고 깊은 이해력을 갖추고 있음을 보여준다.
- 이러한 개선은 UltraChat 데이터 구축 과정으로 인한 향상이며, 대화 생성에서 world knowledge 에 대한 논의를 확장하고 깊이 있게 다룬다. 한편, MMLU에서의 비교적 떨어지는 성능은 특정 분야의 전문 지식 부족을 시사하며, 특화된 Expert LM 을 구축하기 위해 higher quality 의 데이터 생성 기술이 필요함을 시사한다.

# 5.3. Response Quality Evaluation

<span style='color:green;font-weight:bold'> Response Comparison </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/51b02a85-8ee1-4d31-930f-a0912a0d95cd)

- UltraLM은 모든 open-source model 보다 우수한 성능을 나타내며 최대 98% 의 인상적인 win-rate을 보인다.
- UltraLM이 Vicuna보다 9% 더 높은 승률을 기록하는 것도 주목할 만하다.

<span style='color:green;font-weight:bold'> Independent Scoring </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/99670c52-f28d-4fef-8170-850a95815f52)

- Pairwise comparison의 불안정성을 고려하여 GPT-4로 독립적인 품질 점수 산정도 진행한다.
- UltraLM 은 전체 점수 측면에서 모든 open-source model 들보다 현저히 우수한 성능을 보여주며, 이는 각 모델의 성능을 구체적인 유형의 질문과 명령에 대한 인사이트를 제공한다.
- 모든 모델이 commonsense knowledge 와 general world comprehension 에 관련된 간단한 질문에서 더 좋은 성과를 내지만, 추론과 창의적 글쓰기와 관련된 보다 복잡한 작업은 대부분의 어려워한다.

<span style='color:green;font-weight:bold'> AlpacaEval </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/6235cd0c-b2cf-42fd-b1ad-ce7547f32003)

- AlpacaEval leaderboard 에서 text-davinci-003 과의 win-rate 비교에서 4위를 차지한다.

<span style='color:green;font-weight:bold'> Evol-Instruct Evaluation </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/296c37d4-3e7f-4846-9687-d1148d14cc2b)

- Evol-Instruct-test 데이터셋에서 WizarLM 과 비교한다.
- 모든 question 에서 29% 향상이 있으며, WizarLM 이 Evol-Instruct 로 학습된 것을 감안하면 매우 훌륭한 결과이다.

<span style='color:green;font-weight:bold'> Impact of System Prompts </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/74a91d70-9e42-4d41-a9c6-c8a93e6f892f)

- System prompt 를 사용하여 UltraLM의 응답 품질을 향상시킨다.
- 이러한 prompt 는 답변 정확도에 큰 영향을 미치지는 않지만, 주로 정보를 더욱 풍부하게 제공하여 생성된 출력물의 전반적인 품질을 크게 향상시킨다.

## 6. Conclusion
```
In drawing to a close, our work introduces UltraChat, a structured design of multi-turn instructional conversation data primed to foster the growth of general chat models. UltraChat encapsulates a broad range of human-AI interactions, further developing a series of dialogues across various topics and instructions. Statistically, UltraChat shows an impressive presence in critical metrics such as scale, average length, diversity, and consistency, further establishing itself as a leading open-source dataset. We leverage UltraChat to fine-tune the LLaMA model, leading to the development of the robust conversational model, UltraLM. Evaluation across multiple benchmarks reveals that UltraLM surpasses previous open-source models like WizardLM, Vicuna, Alpaca, and Koala in performance.
We eagerly await the innovative research and development that will be catalyzed by our contributions in the field of AI conversational models.
```
