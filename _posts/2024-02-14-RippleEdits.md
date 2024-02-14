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






<span style='color:green;font-weight:bold'> 초록색볼드체 </span>

<span style='background-color: #dcffe4'> 초록색배경 </span>
<span style='background-color: #ffdce0'> 빨간색배경 </span>
