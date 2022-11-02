---
layout: post
title:  "[ICML2022] Dialog Inpainting: Turning Documents into Dialogs"
date:   2022-11-01 10:01:00 +0900
use_math: true
categories: [Dialogue]
---
[[pdf]](https://arxiv.org/pdf/2205.09073.pdf)  &emsp;
[[github]](https://github.com/google-research/dialog-inpainting) <br>

**Zhyun Dai<sup>* 1</sup>, Arun Tejasvi Chaganty<sup>* 1</sup>, Vincent Zhao<sup>* 1</sup>, Adia Amini<sup>1</sup>, Qazi Mamunur Rashid<sup>1</sup>, Mike Green<sup>1</sup>, Kelvin Guu<sup>* 1</sup>**
<br><sup>*</sup>Equal Contribution  &emsp; <sup>1</sup>Google Inc. Mountain View, USA. &emsp; 

![image](https://user-images.githubusercontent.com/42200027/199418938-765bfa4c-7761-42d5-bec9-ba2dcad9bb0e.png)

# Abstract
- 기존 ConvQA 의 scarce training data 문제를 해결하기 위해, document 로 부터 dialogue 를 생성하는 <span style='color:green;font-weight:bold'> dialogue inpainting </span> 방법을 제안한다.
- *dialog inpainting* 방법으로 기존 ConvQA 의 1,000 배 크기의 <span style='color:green;font-weight:bold'> WikiDialog, WebDialog </span> 두 데이터셋을 생성한다.
- 생성한 두 데이터셋을 학습에 활용하여 ConvQA Retreival system 에서 기존 State-of-the-Art model 보다 <span style='background-color: #dcffe4'> 무려 40% 가량의 성능 향상을 보이는 모델 </span>을 제시한다.

# Introduction 
최근 information-seeking tool 들은 잘 정의된 question (ex "Where was Barack Obama born?") 에 대해 좋은 성능을 보이지만, "How to eat healthier?" 와 같 은 conversation 속의 context 와 깊이 있는 이해를 동반한 open-ended 질문에 대해서는 잘 풀지 못한다. 이러한 문제를 해결하기 위해, [ConvQA](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/radlinski2017conversational.pdf) task 가 제안되고 연구가 되고 있다. 그러나 crowdsourcing 의 난이도와 도메인 지식 부족의 이유로 training data 를 구축하는데 비용이 많이 들고 어렵다. 이러한 문제로 현재 ConvQA system task 의 데이터셋들은 대략 10,000 개 정도의 적은 사이즈를 갖게 된다.

한편, Wikipedia, PubMed 와 같이 high-quality document 는 굉장히 풍부하다. 이러한 document 는 전문가들이 작성하는 경우가 많으며, crowdsourcing 으로 얻기도 쉽다. 저자들은 이러한 점에 착안해 이러한 높은 품질의 document 로 부터 dialog 를 만드는 방법을 제안한다. document를 dialog 형식으로 만들기 위해서는, system 이 질문하고 writer 가 답변을 하는 형식으로 진행되는데, writer 의 답변은 document 의 phrase 를 사용하면 되므로 이미 정해져 있다. 이 상황에서 system 이 적절한 question 을 묻도록 만들면 되는데, 이는 마치 옆에서 다른 사람이 전화를 하고 있을 때, 전화 건너 상대방의 말을 유추하는 것과 유사하다. 저자들은 이러한 상황을 <span style='color:green;font-weight:bold'> dialog inpainting </span> 이라고 표현하는데, inpainting 은 computer vision 에서 mask 되거나 오염된 부분을 채우는 방법을 말한다. 

이 ***Inpainter*** 를 통해 저자들은 Wikepedia와 web data 로 부터, <span style='color:green;font-weight:bold'> WikiDialog, WebDialog </span> 을 생성한다. 두 데이텃셋의 크기는 19M+ 으로, 기존의 ConvQA 의 가장 큰 데이터셋보다 1,000배가 큰 사이즈이다. 이후, *conversiotionality* 와 *answer adequacy* 측면에서 생성된 데이터셋을 평가하고, 이 데이터셋을 학습한 모델이 ConvQA system 에 얼마나 좋은 성능을 보이는지 검증한다.

실험 결과, ConvQA retreival benchmark (QRECC, OR-QUAC, TREC-CAST) 에서 <span style='background-color: #dcffe4'> 기존 State-of-the-Art 모델 보다 40% 가량 좋은 성능을 보여주었으며, zero-shot 실험에서도 finetuning 의 95% 에 해당하는 강력한 성능 </span>을 보여준다. 

# Dialog Inpainting
<span style='color:green;font-weight:bold'> Notation </span>

complete dialog $d$ 
$d=(u_1, u_2, ..., u_t, ..., u_T)$ 

unobserved utterances 
$@$ symboal *e.g.* $(u_1, u_2, @, u_4, @)$

shorthand sign that denote dialog $d$ with utterances 3 and 5 masked
$d_{m(3,5)}$

