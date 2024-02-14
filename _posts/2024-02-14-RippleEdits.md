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

## Introduction

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

## Problem Setting

Factual Knowledge | $(e,r,o)$ triple 에 대하여 두 가지 edit type 을 정한다.
_(1) modification_ 은 이미 모델이 가지고 있는 outdated 된 지식 $(e,r,o)$ 를 $(e,r,o*)$ 로 고치는 것이고, _(2) injection_은 새로운 지식 $(e,r,o*)$ 를 주입하는 것이다.

일대일 대응이 되는 (e.g. Date of Birth) injection 의 경우, $(e,r,∅)$ 에서 $(e,r,o*)$ 로 empty objet 를 editing 하는 case 로 볼 수 있다.
반면, Sibling 이나 Occupation 과 같은 one-to-may relation 의 경우, injection edit 이  (e, r, {o1, .., on}) → (e, r, {o1, .., on, o∗}) 로 바꾸는 augment 가 된다.





<span style='color:green;font-weight:bold'> 초록색볼드체 </span>

<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
