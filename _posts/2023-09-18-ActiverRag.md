---
layout: post
title:  "Active Retrieval Augmented Generation"
date:   2023-09-18 16:44:00 +0900
use_math: true
categories: [Retrieval, LLM, PLM]
---

[[pdf]](https://arxiv.org/pdf/2305.06983.pdf) &emsp;
[[github]](https://github.com/jzbjyb/FLARE)

**Zhengbao Jiang <sup>1*</sup>, Frank F. Xu <sup>1*</sup>, Luyu Gao <sup>1*</sup>, Zhiqing Sun <sup>1*</sup>,  Qian Liu <sup>2</sup>, Jane Dwivedi-Yu <sup>3</sup>, Yiming Yang <sup>1</sup>, Jamie Callan<sup>1</sup>, Graham Neubig<sup>1</sup>**
<br><sup>1</sup> Language Technologies Institute, Carnegie Mellon University <sup>2</sup> Sea AI Lab <sup>3</sup> Meta AI Research &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/ef996e6b-21fb-4a80-92ce-c8eca35d26d5)

# Abstract
- (**Hallucination**) 최근 LLM 이 remarkable ability 를 보여주지만, inaccurate output 을 생성하는 hallucination 의 경향성을 보인다.
- (**One-step retrieval and** <span style='color:red;font-weight:bold'> Weakness </span>) 이를 해결하기 위하여 최근 retrieval-augmented LM 이 연구되었지만, 이들은 대부분 단 한 번만 정보를 retrieval 해와 retrieve-and-generate setup 을 구현한다. 이 방법은 정보를 지속적으로 가져와야 할 필요가 있는 long text generation 에 취약하다.
- (**Multi-step retrieval and** <span style='color:red;font-weight:bold'> Weakness </span>) 이에 따라, 다시 여러 번 retreival 을 해와 output 을 생성하는 연구 또한 제안되었지만, 이들은 fixed interval 에 document 를 retrieval 해온다.
- (**Active RAG**) 저자들은 active 하게 <span style='background-color: #dcffe4'> when and what to retrieve </span> 를 결정하는 <span style='color:green;font-weight:bold'> active retrieval augmentated generation </span> 을 제안한다. 
- (**FLARE**) 이를 바탕으로 **F**orward-**L**ooking **A**ctive **RE**trieval (**FLARE**) 를 제안한다. 이는 low-confidence token 에 대하여, 미래에 필요할 정보를 retrieval 해오는 retrieval-augmented generation method 이다.
- (**Experiment**) 4 개의 long-form knowledge-intensive generation task dataset 에 대하여 FLARE 가 superior or competitive performance 를 보여준다.
 
# Introduction
Generative LM ([GPT-3](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html),[instructGPT](https://arxiv.org/abs/2203.02155),[GPT-4](https://arxiv.org/abs/2212.14024),[PAlm](https://arxiv.org/abs/2204.02311),[RAG](https://arxiv.org/abs/2210.01296),[LLama](https://arxiv.org/abs/2302.13971) 등) 는 최근 NLP system 에서 foundamental component 이며 언어를 이해하고 생성하는데 있어서 remarkable ability 를 보여준다. 
LM 이 training 과정에서 엄청난 양의 world knowledge 를 학습하지만, 그들은 여전히 imaginary content 를 생성하는 hallucination 문제가 있다. ([[1]](https://aclanthology.org/2020.acl-main.173/), [[2]](https://aclanthology.org/2021.findings-acl.120/), [[3]](https://arxiv.org/abs/2303.08774))
이러한 hallucination 을 극복하는 방법으로, **retrieval** 을 이용하는 방법에 제안된다. 이 non-parametric retrieval component 를 parametric LM 에 augmenting 하는 방법으로 external knowledge 를 LM 에 부여하는 방법들이 많이 제안되었다.([RAG](https://proceedings.neurips.cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf), [FiD](https://aclanthology.org/2021.eacl-main.74.pdf), [kNN-LM](https://openreview.net/pdf?id=HklBjCEKvH), [Atlas](https://arxiv.org/abs/2208.03299), [ReAtt](https://arxiv.org/pdf/2212.02027.pdf), [REPLUG](https://arxiv.org/abs/2301.12652) 등)

이러한 Retireval-augmented LM 은 보통 *retrieve-and-generate* setup 을 활용하여, user's input 에 기초한 document 를 retrieval 해온 뒤, complete answer 를 generate 한다. 
이러한 <span style='color:red;font-weight:bold'> single-time retrieval-augmented LM </span> 들은 paramter-only LM (no retrieval)의 성능을 크게 뛰어넘었지만, factoid QA 혹은 fact-checking 와 같은 short-form knowledge intensive paradigm 에서만 잘 작동한다. 
이러한 short-form generation 의 특징은 <span style='background-color: #ffdce0'> user's input 에 연관된 정보가 매우 clear 하고, input 에 기반한 relevant knowledge 를 단 한 번만 retrieval 해와도 충분하다는 것 </span> 이다.

최근 long-form QA ([ELI5](https://aclanthology.org/P19-1346/), [ASQA](https://aclanthology.org/2022.emnlp-main.566/)), open-domain summarization, 그리고 CoT 와 같은 long-form output 을 생성하는 능력에서도 LLM 은 좋은 성능을 보여준다. 
이러한 long-form QA 의 특징은  <span style='background-color: #dcffe4'>  answer 를 얻기위한 complex information 들이 input alone 에 항상 evident 하지 않다는 것</span> 이다.
인간이 paper, essay, book 을 쓸 때와 마찬가지로 LM 역시 generation 과정에서 필요한 knowledge 들을 여러번 gathering 해올 필요가 있다. (*would require gathering multiple pieces of knowledge throughout the generation process*)
예를 들어, open-domain summaraization ([[4]](https://arxiv.org/abs/2212.10526)) 에서, initial retreival 은 topic name (e.g. Joe Biden) 에 기반핥테지만, 이들은 모든 aspect 와 detail 을 포함할 수 없다. 
따라서 generation process 중간에 extra-information 을 retrieval 해올 필요가 있다.(e.g the education history of Joe Biden)

이렇게 **multiple time** retrieval 을 해오는 system 을 build 하려는 노력 역시 여러 연구를 통해 존재한다. 
이러한 시도들은 past context 를 *passively* 활용하여, fixed interval 에 additional information 을 retrieval 해온다. ([knn-LM(ICLR2020)](https://openreview.net/pdf?id=HklBjCEKvH), [RETRO(ICML2022)](https://proceedings.mlr.press/v162/borgeaud22a/borgeaud22a.pdf), [RALM](https://arxiv.org/pdf/2302.00083.pdf), [IRCoT(ACL2023)](https://arxiv.org/pdf/2212.10509.pdf))
이 들은 LM 으로 하여금 미래의 generation 과정을 accurately reflect 하거나, inappropriate point 에서 retrieve 해온다.  
몇몇의 work 들은 multi-hop QA 에서 full-question 을 decomposing 한다. ([Self-Ask](https://arxiv.org/pdf/2210.03350.pdf), [ReAct](https://arxiv.org/abs/2210.03629), [DecomP](https://arxiv.org/abs/2210.02406), [DSP](https://arxiv.org/pdf/2212.14024.pdf))

저자들은 follwing question 에 대해서 대답한다 : <span style='background-color: #dcffe4'> can we create a simple and generic retireval-augmented LM that actively decides when and what to retrieve throughout the generation process </span>.
저자들은 <span style='color:green;font-weight:bold'> when to retrieve </span> 를 알아내는 것이 unneccsary or inappropriate knowledge retreival 을 줄이는 과정이라고 설명한다. 
LLM 이 lack of knowledge 에서 low probabilityconfidnce 를 보이고 well-calibrate 를 하려는 시도를 한다는 발견([[6]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00407/107277/How-Can-We-Know-When-Language-Models-Know-On-the),[[7]](https://arxiv.org/abs/2207.05221))에서, 저자들은 <span style='background-color: #dcffe4'> low-probability token 을 LM 이 generate 하려 할 때 retrieval 을 해오는 strategy  </span> 를 택한다.

*What to retrieve* 를 결정할 때는, LM 이 미래에 generate를 하려는 것을 고려하는 것이 매우 중요하기 때무넹, future generation 에 benefit 을 주는 것이 acitve retrieval 의 goal 이다. 
따라서, 저자들은 temporary next sentence 를 생성한 이후에, 이 것을 query 로 하여 relevant document 를 retrieval 해오고, 이후 이 retrieved document 를 활용하여 regenerating 하여 sentence 를 만든다.
이 두 가지 면 (*when and what to retrieve*) 를 반영하여 저자들은 **F**orward-**L**ooking **A**ctive **Re**trieval augmented generation (**FLARE**) 라는 방법론을 제안한다.
<span style='color:green;font-weight:bold'> FLARE iteratively generates a temporary next sentence, use it as the query to retrieve relevant documents if it contains low-probability tokens and regenerate the
next sentence until reaches the end. </span>

FLARE 는 어떠한 LM 에도 적용가능하지만, GPT-3.5 (text-davinci-003)를 활용하여 variety of task 에 적용하였을 때, 매우 좋은 성능을 보여준다 :multihop QA (2WikiMultihopQA), commonsense reasoning (StrategyQA), long-form QA (ASQA) 그리고 open-domain summarization (WikiAsp)

# Retrieval-Augmented Generation
<span style='color:green;font-weight:bold'> Notations and Definitions </span> <br>
<br>
Given user input $x$, document corpus $D$ 에 대하여, retrieval-LM 의 goal 은 $y=[s_1, s_2, ..., s_m] = [w_1, w_2, ..., w_n]$ 을 추출 하는 것이다. ($m$ 개의 문장 혹은 $n$ 개의 token)
Retrieval 을 활용하기 때문에, $y=LM([D_q, x])$ 가 된다. (where $D_q = ret(q)$ with query $q$).

<span style='color:green;font-weight:bold'> Single-time Retrieval-Augmented Generation </span> <br>
<br>
Single-time retrieval-augmented LM 모델은 user input $x$ 를 query $q$ 로 하여, 직접적으로 단 한 번만 retrieval 을 이용한, $y=LM([D_q, x])$ 의 형태가 된다.

<span style='color:green;font-weight:bold'> Activer Retrieval Augmented Generation </span> <br>
<br>
Active RAG 의 formulation 은 다음과 같다.
Step $t$ 에 대하여, retrieval query $q_t$ 는 input $x$ 와 그 전까지 생성된 generated output $y_{<t} = [y_0, ...,. y_{t-1}]$ 에 의존한다. 따라서 query $q_t = qry(x,y_{<t}) 가 된다(where *qry* is the query formulation function).
처음 시작 때는 query 가 input 이다 ($q_1 = x$).
따라서, 최종적으로 output 은 $y_t = LM([D_{q_t}, x , y_{<t}])$ 가 된다.

# FLARE: Forward-Looking Activer REtrieval Augmented Generation
저자들은 두 가지를 가정한다: (1) necessary 정보를 가져올 필요가 없을 때 Retrieval 을 해올 필요가 없으며, (2) future generation 의 intent 를 반영하여 query 가 구성되어야 한다는 것이다. 
이 들을 고려하여 FLARE method 를 제안한다.
[Toolformer](https://arxiv.org/abs/2302.04761) 의 영감을 받아, retrieval query 를 생성하기 위해 LM 에게 instruction prompt 를 부여하는 $FLARE_{instruct}$ 방법과, LM 의 생성결과를 direct search query 로 사용하는 $FLARE_{direct}$ 두 가지 방법이 있다.

<span style='color:green;font-weight:bold'> A FLARE with Retrieval Instructions </span> <br>
<br>
첫 번째 방법은 Toolformer 에서 그러한 것처럼 **"[Search(query)]"** 를 통해 필요한 정보를 retrieval 해오는 것이다. (e.g, "The colors on the flag of Ghana have the following meanings. Red is for [Search(Ghana flag red meaning)] the bloodof martyrs, ...")
GPT-3.5 model 에 few-shot prompting 을 통해 이 행동을 elicit 한다. 

이 행동을 위해 두 가지 스킬이 필요한데, 하나는 seacrh query 를 만드는 skill 을 instruction prompt 로 알려주는 것이고, 다른 하나는 LM 이 answer 를 생성하여 downstream task 를 해결하게 하는 instruction 이다. instruction 에 관한 prompt 들은 아래의 그림과 같이 정리된다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/08814565-0e68-47af-86b3-c208bf60f0e9)

아래의 그림과 같이, LM 이 "[Search(query)]" 를 생성하면, generation 을 멈추고, query term 을 통해 relevant document 를 retreival 해온다. 미래의 user input 전체 prepend 되기 때문에 future generation 에 도움이 된다. 

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/bc049ca7-c57c-4bf8-8383-b2d9cdb40574)

저자들은 LM 이 이 두 가지 skill 을 효과적으로 combine 하여 meaningful 하게 search query 를 생성하고 task 를 수행하는 것을 확인한다.
그러나, 여기에는 두 가지 issue 가 있다: <span style='background-color: #ffdce0'> (1) LM 은 필요한 것보다 적게 search query 를 생성하고, (2) 지나친 (excessive) search query 를 생성하는 것은 answer generation 을 방해하여 perforamnce 에 부정적인 영향을 미친다는 것이다.</span>

각 문제를 해결하기 위해 저자들은 두 가지 방법을 각각 적용했는데, 첫 번째로는 "[" token 의 logit 을 2.0 으로 만들어, "[Search(query)]" 가 최대한 많이 나오게끔 한다.
두 번째로, 한 번 "[Search(query)]" 를 통해 search 가 이뤄진 이후에는 next few token 안에 다시 "[Search(query)]" 가 나오지 않게끔 "[" 에 large negative logit 을 부여한다.


<span style='color:green;font-weight:bold'> Direct FlARE </span> <br>
<br>
$FLARE_{instruct}$ 는 LM 에만 의존하는 방법이므로, black-box LM 을 fine-tune 하지 못한다면, retrieval instruction 을 통해 생성된 query 에 대한 reliablity 를 가질 수 없다. 따라서 저자들은, 직접적으로 retreival 하는 방법론도 제안한다.

<span style='color:green;font-weight:bold'>  Confidence-based Active Retrieval </span> : Figure 1 과 같이, step $t$ 에서, retrieval 과정 없이 temporary next sentence $\hat{s_t} = LM([x,y_{<t}])$ 를 생성한다.
이후 $\hat{s_t}$ 를 통해, retrieval 을 trigger 할지 안할지를 결정한다.
만약 LM 이 $\hat{s_t}$ 에 retrieval 을 통한 additional information 없이도 충분히 confident 하다면, 그대로 문장을 완성한다.
그렇지 않다면, $\hat{s_t}$ 를 통해 $s_t$ 를 재생성(regenerate)한다.
이를 결정하는 것은 threshold $\theta$ 이다.
정리하면 실제 output sentence $y_t$ 는 아래와 같이 생성된다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/b9ea7845-282f-4a85-a96b-c1b560e8f94c)

<span style='color:green;font-weight:bold'>  Confidence-based Query Formulation </span> : 정보 검색을 수행하는 한 가지 방법은 직접 다음 문장 $\hat{s_t}$을 검색 쿼리 $q_t$ 로 사용하는 것이다. 이것은 생성된 hypothetical 제목 또는 단락을 사용하는 기존 방법([[8]](https://arxiv.org/abs/2212.10496),[[9]](https://arxiv.org/abs/2210.01296))과 유사한 접근 방식을 공유한다. 이러한 방법은 원래 입력 질문 대신 언어 모델의 생성물을 검색 쿼리로 사용하는 것이다 ([8], [[10]](https://aclanthology.org/2021.acl-long.316.pdf)). 우리는 이러한 기술을 활용하여 long-form generation 에 적용한다.

Empiricially, next sentence을 사용한 검색이 previous context 을 사용한 검색보다 훨씬 우수한 결과를 얻는 것으로 나타났다(이러한 결과는 6.2 절에서 자세히 설명할 예정이다). 그러나 이것은 그 안에 포함된 <span style='color:green;font-weight:bold'> 오류를 계속 전파할 위험 </span>이 있다. 예를 들어, 언어 모델이 "조 바이든은 펜실베니아 대학에 다녔다"라는 정확하지 않은 정보를 생성하면 올바른 사실인 그가 델라웨어 대학에 다녔다는 대신에 이 오류 포함 문장을 쿼리로 사용하면 검색기가 관련 없는 정보를 검색하게 할 수 있으며, 이는 future generation 을 잘못 이끌 수 있다. 이 문제를 극복하기 위한 두 가지 간단한 방법을 Figure 3 에서 설명하고 있다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/d6196366-c6e0-4fb8-b79f-a60357da5078)

<span style='color:green;font-weight:bold'> Masked sentences as implicit querie </span> : 첫 번째 방법은 $\hat{s_t}$ 내에서 신뢰도가 낮은 토큰을 임계값 β ∈ [0, 1] 아래의 확률로 마스킹 처리한다. 높은 β는 더 강력한 마스킹을 의미하며, 이로 인해 문장에서 잠재적인 산만 요소가 제거되어 검색 정확도가 향상된다.



 
