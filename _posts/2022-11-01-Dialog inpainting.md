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
Dialog Inpainting 을 하기 위해 ***Inpainter*** 를 우선 학습한다. 

<span style='color:green;font-weight:bold'> Notation </span>

complete dialog $d$  <br>
$d=(u_1, u_2, ..., u_t, ..., u_T)$ 

unobserved utterances <br>
$@$ symboal *e.g.* $(u_1, u_2, @, u_4, @)$

shorthand sign that denote dialog $d$ with utterances 3 and 5 masked <br>
$d_{m(3,5)}$
<br><br>
**Inpaint**( $d_{m(3,5)}$ ) = $(u_1, u_2, \hat{u_3}, u_4,\hat{u_5})$

<span style='color:green;font-weight:bold'> Training: Dialog reconstruction </span>

Inpainter training 시 에는 random 하게 하나의 utterance 를 mask 한다. 

$$ d_{m(t)} = (u_1, ..., u_{t-1}, @, u_{t+1}, ..., u_T)$$

이후, [BERT](https://aclanthology.org/N19-1423/) 와 마찬가지로 Maximum Likelihood Estimation(MLE) 방법으로 학습한다.
BERT 에서는 token 하나가 mask 로 되었다면, 이 경우 utterance 하나가 mask 된다.

![image](https://user-images.githubusercontent.com/42200027/199421830-df3f09e8-7a53-4f80-a8e1-1702224391c7.png)

Inpainter 로는 [T5](https://arxiv.org/abs/1910.10683) 가 사용된다. Input 이 하나의 utterance 를 mask 한 text 이고, output 이 하나의 utterance 이기 때문에, text-to-text trasfer transformer 인 T5 가 이상적인 선택이다. 

<span style='color:green;font-weight:bold'> Inference: Transforming documents into dialogs </span>
<span style='background-color: #dcffe4'> Inpainter 의 역할은 document 를 dialog 로 바꾸기 위함이다. </span>
원하는 document 혹은 passage $p$ 가 $(s_1, s_2, ..., s_m)$ 의 문장으로 이뤄져 있을 때, 이 것이 answer 라고 생각하고 question 을 만드는 것이 inpainter 의 역할인 것이다.
따라서 원하는 결과는 $(@,s_1, @, s_2, @, s_3, ..., @, s_m)$ 의 형태의 dialog 이다.
이 때, 저자들은 speaker 에게 부족한 정보를 hint 로 제공하기 위해 prompt 를 앞에 붙여준다. 
prompt 의 형식은 "Hello, I am an automated assistant and can answer questions about (document title)" 이다.

최종적으로, 원하는 parital dialog 는 아래와 같이 입력되어 inpainting 되길 원한다.

$$ ParticalDialog(p) = (s_{prompt}, @, s_1, @, ..., @, s_m). $$

그러나, training 단계에서는 mask $@$를 dialog 당 하나의 utterance 만 학습하도록 하기 때문에, 이렇게는 inference 가 되지 않는다. 
따라서, 저자들은 $(s_{prompt}, @, s_1)$ 에서 inpainting 을 한 번 한 이후, inpainting 된 utterance $\hat{u_1}$ 을 활용하여, $(s_{prompt}, \hat{u_1}, s_1, @, s_2)$ 의 이어지는 inpainting 을 하도록 설계하였다. 이런 식으로 모든 mask 가 채워질 때까지 반복한다. 

<span style='color:green;font-weight:bold'> Applying dialog inpainting to generate an information sekking dialog dataset </span>

저자들은 inpainter 를 학습한 뒤, 위의 Inference 방법으로 dataset 을 생성한다.
Inpainter 의 학습에 사용된 dataset 은 **P**ublicDialog, **T**askMaster, **Q**R-QuAC, 그리고 **Q**ReCC 이다.
이 중, 앞의 두 데이터셋은 양이 많지만, explicit question answering 을 포함하지 않으며, 뒤의 두개는 크기가 작다.
이를 나누어 저자들은 모델을 학습하여, 앞의 두 개만 학습한 $Inpainter_{PT}$, 뒤에 두 개를 학습한 $Inpainter_{QQ}$, 전부 학습한 $Inpainter_{PTQQ}$ 세 모델을 제시한다.
<br>
이후, [Qr-QuA](https://dl.acm.org/doi/10.1145/3397271.3401110)C retrieval corpus 속의 5.9M 크기의 Wikipedia article 과, [Ms Marco](https://arxiv.org/pdf/1611.09268.pdf) retrieval corpus 속의 8.4M 크기의 English web passage 에 inference 방법을 적용하여 dataset 을 생성한다.
Inpainter 모델에 따라 생성되는 dataset 은 $WikiDialog_{PT}$, $WikiDailog_{QQ}$, $WikiDialog_{PTQQ}$, 그리고, $WebDialog_{PT}$ 가 생성된다.

