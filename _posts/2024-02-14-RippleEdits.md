---
layout: post
title:  "[Arxiv 2307] Evaluating the Ripple Effects of Knowledge Editing in Language Models"
date:   2024-02-14 16:35:00 +0900
use_math: true
categories: [Retrieval, LLM, PLM]
---

[[pdf]](https://arxiv.org/pdf/2307.12976.pdf) &emsp;
[[github]](https://github.com/edenbiran/RippleEdits)

**Roi Cohen<sup>1</sup>, Eden Biran<sup>1</sup>, Ori Yoran<sup>1</sup>, Amir Globerson<sup>1,2</sup>, Mor Geva<sup>1,2</sup>**
<br><sup>1</sup> Blavatnik School of Computer Science, Tel Aviv University <sup>2</sup> Google Research &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/3f1c6340-0c2b-4efb-9ffe-527936012f37)


## Abstract
- (**Obsolete knowledge**) 최근 LM 들은 factual knowledge 를 잘 capture 하지만, knowledge 가 obsolete(구식)할 경우, incorrect generation 을 하게 된다.
- (**Existing Edition Evaluation**) 기존에는 이러한 것에 대해서, updated 된 특정 지식을 성공적으로 edit 하는지 평가할 때, individual fact 가 잘 주입되었는지, 그리고 동시에 다른 subject 는 변하지 않았는지 여부를 측정한다.
- (<span style='color:green;font-weight:bold'> Ripple Effect </span>) 이 논문에서는, 하나의 fact에 대한 injection 이 다른 fact 에 대한 update 를 가져 온다는 **"ripple effect"** 를 정의하고 다룬다.
- (**RippleEdits**) 그리고 그 ripple effect 에 대한 criteria 를 정의한 후, 그에 걸맞는 5k factual edit 에 관한 benchmark 인 RippleEdits 를 구성한다. 실험 결과, 여러 모델에서 이러한 ripple effect 를 잘 처리하지 못하는 것을 확인하였고, simple in-context editing baseline 이 좋은 editing 성능을 보임을 확인한다.

## 1. Introduction

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/e82a3a9d-e50f-46cd-be0f-b08794582a9b)

<span style='color:green;font-weight:bold'> ▶ Existing Knowledge Editing (KE) method </span>
<br>

현재 LM 들이 Factual Knowledge 를 잘 capture 하지만, knowledge 가 outdated 될 경우, incorrect factual generation 을 하게 된다.
이를 위해, 여러 연구에서 knowledge editing (KE) 기법을 통해, 이러한 factual error 를 고치려는 시도가 많이 있었따.
Existing KE 방법들은 보통, entity-relation-object $(e,r,o)$ triplet 을 덮어쓰는 (override) 방식으로 (보통, $e$ 와 $r$ 을 덮어쓴다) knowledge editing 을 한다.

이러한 KE 방법들에서 가장 중요한 key point 는 editing success 를 체크하는 "sanity-check" 이다.
보통 $(e,r,?)$ 의 질문을 통해 outdated 된 $o$를 가져오는지 updated 된 $o$를 가져오는지로 평가할 수 있다.
이에 추가적으로 다른 fact 들에 대한 왜곡(distortion)이 있지 않아야 하기 때문에, 그에 대한 평가들도 수반된다.([[1]](https://openreview.net/forum?id=0DcZxeWfOPt),[[2]](https://openreview.net/pdf?id=-h6WAS6eE4),[[3]](https://openreview.net/forum?id=MkbcAHIYgyS)))

<span style='color:green;font-weight:bold'> ▶ Ripple Effects </span>
<br>
이 논문에서는 knowledge editing 이 일어날 때, 어떠한 동반되는 fact 는 같이 변해야하며 (위의 예시에서 messi 가 이적했을 떄, Team 이 가지고 있던 선수 정보에 messi 가 추가되어야 한다), 또 어떠한 사실은 변하지 않아야 한다(메시가 팀을 변경해도 여전히 국적은 아르헨티나이다). 이렇게 하나의 fact 변동이 다른 fact 들에 대해 연동의 결과를 미칠 수 있는 것을 저자들은 <span style='background-color: #dcffe4'> Ripple Effect  </span> 로 정의한다. 이 Ripple Effect 를 제대로 정의하기 위해, 저자들은 여섯 가지 concrete evaluation crieteria 를 제시한다.

<span style='color:green;font-weight:bold'> ▶ RippleEdits Benchmark </span>
<br>
이후, 위의 criteria 들을 기반으로 RippleEdits 라는 benchmark 를 구성한다. 
이는 5k entry 로 이뤄져 있으며, ripple effect 를 고려하여 edit 이 성공적으로 이뤄지는지를 평가한다.
이 benchmark 속에는 timestamp 를 meta data 로 지닌다.

저자들은 이 RippleEdits benchmark 를 활용하여, 5개의 LM 에 3개의 Knowledge Editing 기법을 적용하였을 때, 대부분의 evaluation criteria 에서 poor performance 를 보임을 확인한다.
추가적으로, <span style='background-color: #ffdce0'> (1) larger model 일 수록 ripple effect 를 처리하기 쉬우며, (2) frequent entity 를 edit 하는 것이 logical reasoning error 를 더 많이 발생시킨다 </span> 는 현상을 확인한다.

마지막으로, casual attnetion mechanism 을 기반으로한 <span style='background-color: #dcffe4'> simple in-context editing  </span> 기법을 통해 기존의 parametric KE 방법을 outperform 하는 새로운 방법론을 제안한다.

## 2. Problem Setting

Factual Knowledge | $(e,r,o)$ triple 에 대하여 두 가지 edit type 을 정한다.
_(1) modification_ 은 이미 모델이 가지고 있는 outdated 된 지식 $(e,r,o)$ 를 $(e,r,o*)$ 로 고치는 것이고, _(2) injection_은 새로운 지식 $(e,r,o*)$ 를 주입하는 것이다.

일대일 대응이 되는 (e.g. Date of Birth) injection 의 경우, $(e,r,∅)$ 에서 $(e,r,o*)$ 로 empty objet 를 editing 하는 case 로 볼 수 있다.
반면, Sibling 이나 Occupation 과 같은 one-to-may relation 의 경우, injection edit 이  (e, r, {o1, .., on}) → (e, r, {o1, .., on, o∗}) 로 바꾸는 augment 가 된다.

## 3. Ripple Effects of Factual Edits

전체 knowledge-graph $K$ 에 대하여, edit δ : $(e,r,o) -> (e,r,o`)$ 가 K 속에서 가져오는 변화인 ripple effect 를 $R(δ)$ 로 정의할 수 있다. 그리고 그 크기 $|R(δ)|$ 는 하나의 edit 이 전체 knowledge graph 에 미치는 ripple effect 의 크기로 볼 수 있으며 이를 *severity* 로 정의한다.

# 3.1. Evaluation Criteria
2-hop 내의 ripple effect 를 다음의 6가지로 분류하여 crieteria 를 선정한다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/74c13d38-648f-46c9-a9da-bdfe1eacee1e)

- 각 criteria 에 대한 내용은 논문 참조

## 4. The RIPPLEEDITS Benchmark

# 4.1. Data Generation Pipeline

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/2f15a2ba-68c7-4a39-b82f-6cb73ec8f27c)

<span style='color:green;font-weight:bold'> Step 1: Factual triplets collection </span>
<br>
첫 번째 step 은 fact 를 collection 하는 것이다. 아래의 세 가지 type 을 WIKIDATA 에서 추출한다.
- **RECENT** : 2022 년 7월 이후에 생성된 최신 지식들을 통해 injection editing fact 를 추출

- **RANDOM** : 추후 modification edit 이 될 수 있게 random 하게 fact 를 추출.

- **POPULAR** : Severity 가 큰 경우를 위해 인기있는 triplet 을 추출

<span style='color:green;font-weight:bold'> Step 2: Edits generation </span>
<br>

위의 RECENT 를 기반으로 RANDOM/POPULAR 등의 오래된 지식들을 edit 하는 edit generation 을 진행한다.

<span style='color:green;font-weight:bold'> Step 3: Evaluation tests generation </span>
<br>

새로운 query 에 대해서 이 과정을 반복하여 test set 을 generation 한다.

<span style='color:green;font-weight:bold'> Step 4: Phrasing in natural language </span>
<br>

이후 이 knowledge graph 를 자연어 문장으로 phrasing 한다.

# 4.2. Data Statistics

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/2f5c08a4-9c5e-4965-8cb3-5da849a4a269)

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/a42e14ea-4dd1-4f87-8daf-c8cc236e1a16)

## 5. Experiments

# 5.1. Evaluation Setting

- **Editing Method** : MEND, ROME, MEMIT

- **Baseline** : In-context Editing (ICE)
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/50c473ad-4c72-47e7-b4b7-7070e2b8bc89)

- **Models** : GPT-2 XL, GPT-J, LLaMA, GPT-NeoX, GPT-3

# 5.2. Results

<span style='background-color: #dcffe4'> 결과는 거의 Baseline 실험 제시이다. </span>

<span style='color:green;font-weight:bold'> (1) RECENT </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/a1f31c7b-6094-43b3-98bb-b669d4b25c11)


<span style='color:green;font-weight:bold'> (2) RANDOM </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/e5468948-7471-4bd7-907c-1b76c423ffb7)


<span style='color:green;font-weight:bold'> (3) POPULAR </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/9c2f3522-9883-4cfb-b678-3f3abc540593)


<span style='color:green;font-weight:bold'> (4) Avg. of ROME </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/1f25fb3d-d607-4828-b962-79ffc3840ba9)


<span style='color:green;font-weight:bold'> (5) Accuracy of MEND, ROME, MEMET </span>
<br>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/2bc5b080-c395-4529-a2c9-1caceb4949ee)

## Conclusion and Disccusion
```
We introduce the notion of ripple effects in knowledge editing, suggesting that editing a particular fact implies further updates of related facts. We additionally propose evaluation criteria for ripple effects and create RIPPLEEDITS, a diagnostic benchmark designed to evaluate how well KE methods handle the ripple effects of various edits. We evaluate prominent KE methods and show that they often fail to introduce consistent edits that capture the ripple effects of an edit, suggesting that future development of KE methods should consider those effects more carefully. Last, we show that a simple in-context editing method achieves the best results on RIPPLEEDITS, highlighting the potential of such editing approaches.

Notably, our benchmark covers a small fraction of all possible ripple-edits. For example, one could consider ripple effects that involve more than two hops, and explore the graph structure of different edits. In addition, while we focus on ripple effects of single edits, future work can consider the effect of editing multiple facts in a single batch. Finally, it would be interesting to consider cases where models succeed in capturing ripple-edits, and analyze how these are implemented mechanistically in the transformer architecture.
```
