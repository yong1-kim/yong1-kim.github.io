---
layout: post
title:  "[ICML2022] Describing Differences between Text Distributions with Natural Language"
date:   2022-11-12 17:38:00 +0900
use_math: true
categories: [LLM, Transformer]
---
[[pdf]](https://proceedings.mlr.press/v162/zhong22a/zhong22a.pdf)  &emsp;
[[github]](https://github.com/ruiqi-zhong/DescribeDistributionalDifferences) <br>

**Ruiqi Zhong<sup>1</sup>, Charlie Snell<sup>1</sup>, Dan Klein<sup>1</sup>, Jacob Steinhardt<sup>1</sup>**
<br><sup>1</sup> Computer Science Division, University of California, Berkeley. Correspondence to: Ruiqi Zhong  &emsp; 

![image](https://user-images.githubusercontent.com/42200027/201470594-acfa0565-f6df-44a6-88d6-4d92c3be2f60.png)

# Abstract
- (Motivation) 두 text 의 *distribution* 이 다르다는 것을 어떻게 알 수 있을까? 인간은 많은 sample 을 직접 읽는 과정이 필요하므로 굉장히 많은 시간이 걸린다. 
- (Solution) 이 논문에서는 **GPT-3 를 활용하여 automatically describe distribution of text** 방법을 제안한다. 이 방법은 기존에 없던 새로운 framework 로 다른 다양한 task 에도 적용 가능하다. 
- (Method) "[samples of $D_0$] + [samples of $D_1$] + the difference between them is __." 의 prompt 를 이용하여 GPT-3를 fine-tuning 시킨 뒤, 생성되는 decription sentence 를 각 dataset 에 얼마나 matching 되는지로 reranking 한다. 
- (Result) 기존 GPT-3 Curie (13B) 모델은 human annotation 과 7% 의 유사도를 보이지만, fine-tuning 이후 61% 로 증가하였으며, GPT-3 Davinci (175B) 모델을 활용했을 때는 76% 나 올라, **제안한 방법으로 생성한 description 이 text distribution 을 잘 표현함**을 실험적으로 증명한다. 

# Introduction
"What inputs trigger a neuron in my deep learning model? How are the train and test distributions different for my application? How did public opinions on Twitter change from last year to this year?" 이러한 질문들을 생각했을 때, 인간이 이러한 new pattern 을 발견하는 것은 많은 sample 을 직접봐야하고 intractable 하다. <span style='background-color: #dcffe4'>본 논문에서는 두 distribution 사이의 difference 를 발견하고, 그 difference 를 자연어 문장으로 describe 하는 방법</span>을 제안한다. 

제시하는 방법론은 **Learning a natural language hypothesis** 라는 방법으로, two text distribution $D_0$ 와 $D_1$ 에 대해서, $D_0$ 보다 $D_1$ 을 더 잘 설명하는 natural language hypothesis $s$ 을 찾아내는 방법이다. 위의 그림과 같이, $D_0$ 와 $D_1$ 이 있을 때, *"is military-realted"* 라는 문장이 $D_0$ 보다 $D_1$ 을 설명할 hypothesis 이다. 또 추가적으로, $D_0$ 를 train set 으로, $D_1$ 을 test set 으로 놓음으로써, **train-test set distribution difference 를 설명하는 자연어 문장**을 만들 수 있다. 그 예시는 *"is longer in sentence length"* 등이 있다. 그리고, 마지막으로 public opinions shift 에 대해서도 적용 가능하며 그 예시로 *"is optimistic about the pandemic."* 등이 있다.

![image](https://user-images.githubusercontent.com/42200027/201511601-e3f72c04-92bc-4a51-a6ad-6323c6d9e3f5.png)

이 방법론은 GPT-3 Davince 에 prompt 를 이용하여 hypotheses $s$를 생성한다. 하지만 GPT-3 는 limited context size 를 갖고 있기 때문에, 이러한 prompt 는 단지 몇 가지(few) sample 만을 담을 수 있고, whole distribution 은 담을 수 없다. 따라서, 저자들은 re-ranking 방법을 통해, candidate 들이 larger set of sample 에 대해서 얼마나 잘 설명할 수 있는지 확인하는 verifier 를 도입한다. 이에 대한 설명은 위의 그림에 나와있다. 

![image](https://user-images.githubusercontent.com/42200027/201511661-83f08a52-1d04-4d4b-a6cb-65b435556716.png)

그리고, GPT-3 는 hypothesis 를 propose 하는데 최적화 되어있지 않기 때문에, fine-tuning 을 통해 더 발전될 수 있다.
하지만 이러한 task 를 위한 corpus 가 존재하지 않으므로, 저자들은 GPT-3 를 이용하여 data 를 collection 하여 fine-tuning 에 사용한다.
위의 그림과 같이, hypothesis $s$ 에 대해, GPT-3 를 활용하여 sample 들을 generation 한 이후, 그 것들을 human 이 annotate 하여, proposer fine-tuning 에 활용한다. 

[54 real-world binary classification datasets](https://aclanthology.org/2021.findings-emnlp.244/) 에 대해서 검증을 진행한다. 이 dataset 들은 positive class 들에 대해 자연어 description 으로 annotate 되어 있다. 이 문제로 적용을 위해, positive/negative class input 들을 $D_1$/$D_0$ 로 여기고, top-5 description 이 human annotation 과 일치하는 지 비교한다. GPT-3 Curie (13B) 모델을 적용했을 때는 7%의 일치도를 보였지만, fine-tuning 이후 61% 의 일치도를 보여 크게 향상되었고, GPT-3 Davinci model 을 했을 때는 76% 에 도달하였다.

![image](https://user-images.githubusercontent.com/42200027/201511897-6326947c-2269-45c8-98a7-28a73aef7751.png)

이후, 저자들은 기존 존재하던 classification dataset 들이 자신들이 제안하는 시스템의 desciption 과 agree 하는지 실험을 진행한다. 
이 시스템은 subjectivity analysis 에서 [SUBJ dataset ](https://aclanthology.org/P04-1035/), 데이터셋이 movie review와 plot summary 를 contrast 하는 것으로 구성이 되어있음을 recognize 했지만, 많은 연구에서 이러한 점을 모른 채 zero/few-shot dataset 으로 활용하고 있다고 지적하고 있다.
<span style='background-color: #dcffe4'> 그리고 제안된 시스템은 여러 데이터셋들의 단점을 지적하고 있다. </span>
예를 들어, [MNLI](https://aclanthology.org/N18-2017/) 에서 "contradiction class" 에 "negation" 이 spuriously 관여하고 있으며, [SMS Spam classification dataset](https://dl.acm.org/doi/abs/10.1145/1166160.1166191) 의 경우, spam 으로 분류된 것들은 **항상 hyperlink** 를 포함하고 있음을 발견했다. <span style='background-color: #dcffe4'> 그리고, 이 시스템은 text clustering 에도 사용될 수 있다.</span>

# Learning a Natural Language Hypothesis
X 를 set of all text input 이라고 하면, natural language hypothesis $h$ 는 string $s$ 에 parameterized 되고, 다음과 같이 two input 을 boolean 으로 mapping 한다.

![image](https://user-images.githubusercontent.com/42200027/201512293-2207804b-a03d-4165-97ae-051d7d61a7fd.png)

where $h_s(x_1,x_0) = 1$ means $x_1$ is more $s$ than $x_0$.
예를 들어, $s$ 가 *"is longer in sentence length"* 일 때, $h_s(x_1,x_0) = 1$ 은 $x_1$ 이 $x_0$ 보다 길다는 것을 의미한다.
정리하면, $h_s$ 의 semantic 은 

![image](https://user-images.githubusercontent.com/42200027/201512363-64880c33-683d-4f82-b289-fca2979d4553.png)

으로 정리할 수 있다.
$D_0$ 와 $D_1$ 이 X 의 두 distribution 이라고 하고, $H$ 를 $h$ 의 space 라고 했을 때, 이 task 의 목적은 $H$ 속의 $h$ 중 다음의 "classification accuracy" CA 가 높은 것을 찾아내는 것이다.

![image](https://user-images.githubusercontent.com/42200027/201512430-d13fc2f2-f66e-4873-b9cb-1de3f9c91bbb.png)

식에 대해서 잠시 살펴보면, 두 distribution $D_0$ 와 $D_1$ 으로 부터 뽑힌 sample 들에 대해, $h$ 가 어디로 부터 와쓴ㄴ지를 classify 하는 기존의 statistical machine learning 과 같다. 하지만, traditional statistical machine learning 과 다르게, 이 문제는 두 가지 문제를 가지고 있는데, 첫 번째는 **Search** 문제로, discrete string space 에서 hypothesis 를 searching 하는 것은 어렵다는 것이다. 그리고 두 번째는 **Verify** 문제로, $h_s(x_1,x_0)$를 계산하는 데는 human annotation 이 필요한데, 이 것으 매우 비싸다. 이 연구에서는 neural network 로 human response 를 approximating 하는 방법에 대해서 다룬다.

# Method
