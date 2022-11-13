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

식에 대해서 잠시 살펴보면, 두 distribution $D_0$ 와 $D_1$ 으로 부터 뽑힌 sample 들에 대해, $h$ 가 어디로 부터 지를 classify 하는 기존의 statistical machine learning 과 같다. 하지만, traditional statistical machine learning 과 다르게, 이 문제는 두 가지 문제를 가지고 있는데, 첫 번째는 **Search** 문제로, discrete string space 에서 hypothesis 를 searching 하는 것은 어렵다는 것이다. 그리고 두 번째는 **Verify** 문제로, $h_s(x_1,x_0)$를 계산하는 데는 human annotation 이 필요한데, 이 것으 매우 비싸다. 이 연구에서는 neural network 로 human response 를 approximating 하는 방법에 대해서 다룬다.

# Method
![image](https://user-images.githubusercontent.com/42200027/201512602-6e130777-5e1a-4995-8fc6-0449d5370a33.png)
본 논문에서는 GPT-3 를 prompt 하여 small set 에 대해 hypothesis 를 만들고(1), UnifiedQA 를 통해 larger set 에서 hypothesis 를 검증하고(2), data collection pipeline(3) 을 통해, proposer 와 verifier 를 fine-tuning(4) 한다. 이 과정들은 위의 그림에 요약되어있으며 하나씩 차례대로 살펴본다.

<span style='color:green;font-weight:bold'> (1) Hypothesis Proposer </span>
<br>
![image](https://user-images.githubusercontent.com/42200027/201512732-d7699698-64f5-47e3-8d3c-0096544c3ab9.png)

저자들은 GPT-3 를 이용하여 hypothesis 를 생성한다. 그림과 같이 $D_1$ 으로부터 몇 개의 sample을, $D_0$ 로 부터 몇 개의 sample 을 추출 하고, *"Compared to group 0, each sentence from group 1 ___"* 이라는 prompt 를 집어 넣어준다.
GPT-3 는 2048 의 context size limit 이 있기 때문에, 각 sample 크기는 5 개이다.
Controlled decoding 기법이 없으면, prompt completion 이 *"is more positive, while sentences from group 0 are ungrammatical."* 과 같이 나타난다.
그러나, 이러한 completion 은 undesirable 한데, verifier 가 한 번에 두 개 (positive, ungrammatical) 을 확인해야 하고, 두 번째 hypothesis 는 group 을 평가해야 하는데, verifier 는 sample 들을 평가할 수만 있기 때문이다.
<span style='background-color: #dcffe4'> 따라서, 저자들은 GPT-3 가 "group" 이라는 token 을 decode 하는 것을 막고, "," 와 "or" 같은 token 을 생성하는 것을 금지시킨다.</span>

그리고, $D_0$ 와 $D_1$ 이 완전히 같거나 많이 유사할 경우, optimal hypothesis $h^\*$ 는 이들을 잘 구분할 수 없어야 한다.
그러나, 몇 가지 sample 을 뽑아서 GPT-3 를 prompt 할 경우에는 이 optimal hypothesis 를 만족시킬 수 없으므로, proposer 를 혼동시킬 수 있다. 
이 것을 막기 위해 저잗르은 [RoBERTa-Large](https://arxiv.org/abs/1907.11692) model 을 학습시켜, 각 sample 이 $D_0$ 와 $D_1$ 중 어디서 오는지 예측하게 한 다음에, confidence score 를 기준으로 top-$p$ group 을 만든다.
실험에서는 top-5, top-20, top-100 group 에서 각각 10 번씩 sample 들을 뽑은 후, 2 개의 completion 을 만들게 하여 최종적으로 3 x 10 x 2 = 60 의 hypotheses 를 얻고, 이를 re-rank 한다.

<span style='color:green;font-weight:bold'> (2) Hypothesis Verifier </span>
<br>
위의 CA 수식을 검증해야하는데, $h_s(x_1,x_0)$ 는 expensive human annotation 이 필요하기 때문에, neural network 를 이용하여 approximation 한다.

![image](https://user-images.githubusercontent.com/42200027/201513189-a09a4c2b-7d28-4907-8749-c314fe61d5a8.png)

neural network $V$ 에 대해, $V(s,x_1,x_0)=1$ 은 $x_1$ 이 $x_0$ 보다 더 $s$ 하다는 것을 의미한다. 

이후, 저자들은 [UnifiedQA](https://aclanthology.org/2020.findings-emnlp.171/) 를 verifier 로 활용한다. 이 것은 [T5](https://arxiv.org/abs/1910.10683) model 을 기반으로 한 Question answering 모델이다. 

![image](https://user-images.githubusercontent.com/42200027/201513277-cb6cea98-79ea-493e-85e2-c3fd4e40c017.png)

위의 그림과 같이, context $c$ 는 pair of sentence A from $D_1$, and sentence B from $D_0$ 이다. quesiont $q$ 는 *"is it true that sentence A is more positive?"* 이고, *"is more positive"* 부분은 hypothesis $s$ 이다. 이후, 이것을 QA 모델인 UnifiedQA 에 돌렸을 때 1 이 나오면 "yes", 아니면 "no" 가 나온다. 
이후, $V(s,x_1,x_0)$ 값을 통해 CA 를 re-ranking 한다. 
전부 re-ranking 하지는 않고, 400 개의 random $(x_1,x_0)$ sample 에 대해서만 $V(s,x_1,x_0)$ 값을 구하고, 최종적으로, 5 개의 hyphothesis $s$ 를 남긴다.

<span style='color:green;font-weight:bold'> (3) Collecting Data for Supervision </span>
<br>

(1) 에서 proposer 로 사용된 GPT-3 와 (2) 에서 verifier 로 사용된 unifiedQA 모두 이 태스크를 위해 학습된 것이 아니기 때문에 최적화되어 있지 않다. 따라서 fine-tuning 을 통해 그 성능을 향상 시킬 수 있다. 
그러나 이러한 태스크를 풀기 위한 corpus 가 없기 때문에 fine-tuning 을 진행할 수 없기 때문에, new dataset 을 collect 한다.

Proposer 의 fine-tuning 을 위해서는 more $s$ 스러운 5 개의 sample, less $s$ 스러운 5 개의 sample 이 있어야 하고, verifier 를 fine-tuning 하기 위해서는 $x_1$ 이 $x_0$ 보다 더 $s$ 스러운 triplet $(s,x_1,x_0) 가 필요하다.
이를 위해서 저자들은 특정 hypothesis $s$ 에 대해, GPT-3 에 $s$ 를 만족하는 sample 과 그렇지 않은 sample 들을 생성시키게 하였다.

<span style='color:green;font-weight:bold'> Curating Hypothesis </span>
<br>
![image](https://user-images.githubusercontent.com/42200027/201513759-4cbfd64b-d161-42f6-83fb-07f079283a6d.png)

첫 번째로, hypothesis 를 여러개 추출하기 위하여, [GPT-3](https://arxiv.org/pdf/2005.14165.pdf)의 도움을 받는다. 
자세히는 <span style='background-color: #dcffe4'> 10개의 hypothesis 를 직접 생성한 후, GPT-3 에게 "brainstrom 해" 라는 prompt 를 활용해 생성</span>하였다. 생성되는 hypothesis 는 shallow (e.g. "contains the word 'yay' at the ned of the sentence') 한 것부터, topical ("loves school")한 것, 그리고 social and linguistic cue 를 다루는 complex 한 것("supports universal healthcare", "is written in first person")까지 다양하다. 

<span style='color:green;font-weight:bold'> Conditional Generation </span>
<br>
![image](https://user-images.githubusercontent.com/42200027/201513908-da6121fb-22c3-4c3b-aa5b-a955eb982734.png)

hypothesis $s$ 가 "love school" 이라고 했을 때, positive sample 은 "My advisor is really helpful and I learned a lot" 등이 있다. 모델을 fine-tuning 하기 위해서는 positive sample 과 negative sample 이 모두 필요하다.
<span style='background-color: #dcffe4'> Positive sample 을 생성하기 위해 위의 그림처럼 GPT-3 모델을 활용한다.</span>  가끔 $s$ 가 "love school" 인데, *"I love school"* 과 같이 겹치는 문장이 생성될 수 있어, $s$ 에 나오는 token 을 생성하지 않게 막아놓는다.

<span style='background-color: #dcffe4'> Negative sample 을 생성하기 위해서 다른 hypothesis 의 positive sample 을 이용한다. </span> 
*"talks about microwaves"* 와 같은 highly-specific 한 예시에 대해서는, 다른 아무 hypothesis 의 positive sample 이 negative sample 이 될 수 있다.
그러나, *"uses past tense"* 와 같은 경우, 직접 contrast hypothesis 인 *"uses future tense"* 를 만들었다. 이렇게 expanded hypothesis pool 이 352 개로 늘어났고 (기존 300개), 이 것들을 이용하여 15 postive sample 들을 만들어 negative sample 로 활용한다.

<span style='color:green;font-weight:bold'> Verifying with Human Annotations. </span>
<br>
![image](https://user-images.githubusercontent.com/42200027/201514151-7e7c2b5b-15a6-4d95-bfda-3162db777e92.png)

instruct GPT-3 의 성능이 매우 좋지만, reliability 를 위해 human tucker 를 활용하여 verify 한다. 
이 Majority vote 를 통해 302 hypothesis 와 각각에 대응하는 5개의 positive/negative sample 들이 남았다.

<span style='color:green;font-weight:bold'> Fine-tuning</span>
<br>
Proposer fine-tuning 을 위해 302 개의 hypothesis 에 대해서, positive/negative sample 5 개 씩 주어주고, hypothesis 를 generate 하게하여 GPT-3 를 fine-tuning 하였다. 2 epoch 을 돌리고, 20 batchsize, 0.05 의 learning rate 를 사용하였다.

Verifier fine-tuning 을 위하여, $V(s,x_1,x_0)=1$ 이 되게, $V(s,x_0,x_1)=0$ 이 되게 하여 unifiedQA 를 fine-tuning 하였다.
하나의 $s$ 당 30 개의 $(x_1,x_0)$ pair를 생성하였고, 250 step, batchsize 32, lr 5e-5 를 사용하였다.
out-of-distribution robusteness 를 위해, 기존 unifiedQA 의 weight 과 fine-tuned unifedQA weight 을 average 하였다.([[1]](https://arxiv.org/pdf/2109.01903.pdf))

# Benchmarking Performance
<span style='color:green;font-weight:bold'> Dataset </span>
<br>
저자들의 [previous paper](https://aclanthology.org/2021.findings-emnlp.244.pdf) 에서, 54 개의 binary text classification task 에 대해 positive class 에 하나 이상의 자연어 description 이 있는 eval set 모음을 차용한다.
이 eval set 들에는 topic classifciation, grammaticallity classifciation, stance classification 등을 포함한다.
각각에 대하여, 제안된 시스템에 postiive class sample 들이 negative class sample 들과 어떻게 다른지를 설명하도록 시키고, human annotation 과 top-5 비교를 한다. human annotation description 을 위한 $s^\*$ 를 "correct" 라고 가정한다.

<span style='color:green;font-weight:bold'> Evaluated Systems. </span>
<br>
larger proposer, a fine-tuned proposer, and a verifier for re-ranking 의 세 요소를 모두 갖추면 description generation 성능이 올라갈 것이라고 추측한다. 
따라서 저자들은 "(1) : Our best system which use fine-tuned GPT-3 Davinci (175B) as the proposer, (2) : a smaller proposer size (fine-tuned Curie, 13B), (3) : no fine-tuning (zero-shot Curie 13B), (4) : no fine-tuning (zero-shot Cuire, 13B) + no verifier for re-ranking, (5): "memorization proposer", where the proposer only generates the hypothesis we curated" 라는 5 개의 모델을 제시하고, 그들의 가정이 맞다면, <span style='background-color: #dcffe4'> (1)>(2)>(3)>(4), 그리고 (2)>(5) </span> 가 될 것이라고 추측한다.

<span style='color:green;font-weight:bold'> Automatic Evaluation. </span>
<br>
Automatic metric 으로는 [BERTScore](https://arxiv.org/abs/1904.09675) 를 활용한다. 
Human annotation 와 top-5 description pair 들을 BERTScore 로 계산한다. 
54 개의 task 에 대해 average 한 후, 5 개의 top-5 중 가장 높은 pair 를 선택한다.
그 결과, (1) : 0.930, (2) : (0.927), (3) : 0.907, (4) : 0.899, (5) : (0.916) 으로, 저자들이 추측한 결과가 나왔다. 하지만. 이 결과들이 모두 높게 측정이 되었기 때문에, manual evaluation 을 추가적으로 진행한다. 

<span style='color:green;font-weight:bold'> Manual Evaluation. </span>
<br>
![image](https://user-images.githubusercontent.com/42200027/201515540-46a24ff2-0c35-4af0-b959-cc6362e56edb.png)

사람들에게 위와 같이 평가해달라고 했을 때, 아래와 같이 모델들에 대해서 결과가 나왔다.

![image](https://user-images.githubusercontent.com/42200027/201514874-1052138f-0fce-4e07-8178-8d518f579349.png)

(4)번 모델 GPT-3 Curie(no fine-tuning proposer + no re-ranking model) 은 (A)+(B) 평가에서 7% 의 human annotation 과 일치하지 않았지만 (4/54), (2)번 모델은 GPT-3 Curie의 proposer fine-tuning  을 통해 61% 일치도 (33/54), (1)번 모델은 GPT-3 Davinci proposer fine-tuning 를 통해 76% 의 일치도(41/54)를 보인 것을 확인할 수 있다.

<span style='color:green;font-weight:bold'> Comparing Verifiers. </span>
<br>

<span style='background-color: #dcffe4'> 저자들은 verifier 가 실제로 효과적인지 실험적으로 검증할 수 없었다고 한다.</span> 그러나, repeatedly 반복되는 hypothesis 를 verifier 가 제거해주는 효과가 있다고 한다. Verifier 를 비교하기 위해, 앞에서의 CA 수식에 대하여, 저자들은 larger and fine-tuned verifier 가 더 좋을 것이라고 추측한다.

![image](https://user-images.githubusercontent.com/42200027/201515727-87b0c7fe-e554-44a6-89a1-4eaa22cecc81.png)

결과는 위와 같은데, CA 수식은 여전히 approximation 이므로 automatic evaluation 은 infeasible 하지만, unifedQA 가 verifier 로서의 역할을 하고, fine-tuned verifier 의 효과가 더 좋았다. 그리고, 실제 [state-of-the-art model](https://arxiv.org/pdf/2112.11446.pdf) 은 unifiedQA 보다 25x 크기 때문에, 그래프의 해석대로라면 훨씬 더 좋은 성능을 보일 수 있다.

# Application
본 연구의 시스템은 suumarize training task, debug dataset shortcut, describe distribution shift, 그리고 label text cluster 에 사용될 수 있다.

<span style='color:green;font-weight:bold'> Summarizing Training Tasks </span>
<br>
