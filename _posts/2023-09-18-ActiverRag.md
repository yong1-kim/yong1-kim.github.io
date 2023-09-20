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

<span style='color:green;font-weight:bold'> Masked sentences as implicit querie </span> : 첫 번째 방법은 $\hat{s_t}$ 내에서 신뢰도가 낮은 토큰을 임계값 β ∈ [0, 1] 아래의 확률로 마스킹 처리한다. 높은 β는 더 강력한 마스킹을 의미하며, 이로 인해 문장에서 잠재적인 distraction 요소가 제거되어 검색 정확도가 향상된다.

<span style='color:green;font-weight:bold'> Generated questions as explicit queries </span>: 다른 방법은 $\hat{s_t}$ 의 확신이 낮은 span 을 대상으로 명확한 질문을 생성하는 것이다. 예를 들어, 만약 LM 이 '펜실베니아 대학교'에 대해 확신하지 못한다면, '조 바이든은 어떤 대학을 다녀왔나요?'와 같은 질문은 관련 정보를 검색하는 데 도움이 될 수 있다. Self-ask ([Press et al., 2022](https://arxiv.org/abs/2210.03350)) 는 이를 수행하기 위해 프롬프트 4.1 (뒤에 등장)에서 나중에 나오는 downstream task exemplar 에 직접 follow-up 질문을 수동으로 삽입하는 방식으로 이루어져 있으며 이는 작업 additional annotaion 을 필요로 한다. Specifically, 저자는 추가적인 어노테이션 없이 낮은 확신 스팬에 대한 질문을 생성하는 범용적인 방법을 개발했다. 구체적으로, $\hat{s_t}$에서 β 아래의 확률로 모든 span을 추출한 다음 각 추출된 span $z$에 대해 답할 수 있는 질문 $q_{t,z}$를 생성하도록 GPT-3.5-turbo에 프롬프트를 지시한다. 프롬프트는 아래와 같다.
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/b2a803f2-3a5c-47aa-8374-22bfccccfc7f)

이후 저자들은 generated question 과returend document 를 통해 answer 를 생성한다. 정리하면 $\hat{s_t}$ 를 위한 $q_t$ 는 아래와 같다.
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/7b755c8c-6c62-4c63-825b-73ef4c82f6a1)

# Implementation Details
Method 검증을 위해, GPT-3.5 LM 인 text-davinci-003 을 이용하여 API 를 반복적으로 query 하여 확인한다. 

**Inital qeury**
시작 query 는 FLARE 가 user input $x$ 를 통해 문서를 검색하고, 첫 번째 문장인 $\hat{s_1} = LM([D_x, x])$ 를 생성하여 반복적인 생성프로세스를 시작한다. 

**Sentence tokenization**

각 step $t$ 마다 대부분의 문장보다 긴 64개의 토큰을 생성하고, NLTK 문장 토크나이저를 사용하여 첫 번째 문장을 추출하고 나머지는 삭제한다. 
 
 **Document corpus and retrievers**
이 연구에서는 retrieval과 generation의 통합에 중점을 두고 있기 때문에, 입력으로 query를 받고 relevant document list 를 반환하는 off-the-shelf retriever를 사용한다. Wikipedia에서 지식을 주로 활용하는 데이터셋의 경우, [Karpukhin et al. (2020)](https://aclanthology.org/2020.emnlp-main.550/)의 Wikipedia 덤프를 사용하여 문서 코퍼스로 사용하며, 문서는 100-토큰 단위로 분할되고 BM25 ([Robertson and Zaragoza, 2009](https://www.nowpublishers.com/article/Details/INR-019))를 retriever로 사용한다. Open-web 에서 지식을 활용하는 데이터셋의 경우, Bing 검색 엔진을 retriever 로 사용한다.

**Retrieved document formatting**
Multiple retrieved document 는 그들의 순위에 따라 linearized 되어 user input 의 시작부분에 다음 형식으로 추가된다:

 ![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/76c592e9-7b98-4210-b001-cce44ee4e792)

# Multi-time Retrieval Baselines
기존의 passive multi-time retrieval augmented LM 들 역시 FLARE framework 를 사용하여 formulate 될 수 있다. 이 연구에서는 세 가지 baseline category 를 introduce 한다. 이 baseline 은 이전 작업들이 다양한 디자인 선택을 가져가기 때문에, 직접적인 비교가 불가능하기 때문에 공식적인 reproduction 결과는 아니다. 저자들은 관련 없는 디자인을 제외하고 동일한 설정을 사용하여 구현되도록 하고, 유일한 차이점은 **when and what to retrieve**이다.

**Previous-window** 는 모든 직전의 $l$ 개의 token 을 query 로 사용한다. RETRO 와 IC-RALM, 그리고 KNN-LM 이 여기에 속한다. (KNN-LM 의 경우 모든 token 에 대해 retrieval 진행)

**Previous-sentence** 는 모든 sentnece 에서 retrieval 을 진행한다. IRCoT 가 여기에 속한다.

**Question decomposition** 은 LM 으로 하여금 sub-question 으로 decompose 하여 question 을 여러 query 로 나눠서 retireval 하게 한다. Self-ask 가 이러한 category 에 속하며, 아래의 prompt 를 통해 이뤄진다:
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/b37a34d8-0000-448e-9706-cd419530bffc)

위에서 언급한 세 가지 method 들은 모두 generation process 에서 additional information 을 검색할 수 있다. 그러나 그들은 notable drawback 을 가지고 있다: (1) Fixed interval approach 는 이전에 생성된 token 을 query 로 사용하며, 이는 LM 이 미래에 생성하려는 내용을 반영하지 못할 수 있다. (2) Fixed interval 에서 정보를 검색하는 것은 부적절한 시점에서 발생할 수 있기 때문에 비효율적일 수 있다. (3) Query decomposition 방법은 task-specific prompt engineering 이 필요하며, 이는 새로운 task 에서의 generalization 이 제한된다. 

# Experimental setup
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/48a84861-3ae9-4765-8fbd-ed3314a6dbb9)

FLARE 의 효과를 검증하기 위해 저자들은 few-shot in-context learning (ICL) 을 사용하여 4 가지 task 에 적용한다. Fair comparison 을 위하여, FLARE 의 결과를 동일한 setting, 즉 동일한 context exemplar, prompt format, retriever, 그리고 document corpus 에서 비교한다. Cost 문제로, 각 데이터셋에서 최대 500 개의 예시를 하위 샘플링하는 [IRCoT](https://arxiv.org/abs/2212.10509) 방법을 따른다. FLARE 의 hyper-parameter 는 dev set 을 통해 선택되며 아래 표와 같다. 특별히 명시되지 않는한, FLARE 는 $FLARE_{direct}$를 나타낸다. Previous-window approach 의 경우, [Ram et al.2023](https://arxiv.org/abs/2302.00083) 을 따라 $l=16$ 의 window size 를 사용한다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/89556ea2-455e-4a6a-bb87-502bbbadbb6a)

[Dataset 설명은 생략]

# Experimental Results
<span style='color:green;font-weight:bold'> Comparison with Baselines </span><br>
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/881759bf-069a-4739-8dd9-09c3ac1e4af3)

여러 task 와 datset 중 multihopQA 에서 눈에 띄는 향상이 보인다. 이는 주로 task의 명확한 정의와 final answer 을 2 단계 추론 과정을 통해 생성해야하는 구체적인 목표 때문에, LM이 주제에 관련된 결과물을 생성하기가 더 쉬워지기 때문이다. 이와 대조적으로, ASQA와 WikiAsp는 덜 명확하게 정의되어 있으며 더 개방적(open-ended)이며, 이는 생성과 평가의 어려움을 증가시킨다. ASQA-hint의 개선은 ASQA보다 큰데, 모호한 측면을 식별하는 것은 많은 경우에 인간에게도 어려운 일이며, 일반적인 힌트를 제공하면 LM이 주제를 유지하는 데 도움이 된다. 

<span style='color:green;font-weight:bold'> Thorough comparisons with baselines </span><br>
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/19cdfc25-ab86-425d-af50-98d483bb4f04)

2WikiMultihopQA에 대한 모든 baseline 성능은 Table 1에서 볼 수 있다. FLARE은 모든 베이스라인 대비 큰 차이로 우수한 성능을 보이며, 이는 미래를 내다보는 액티브 검색이 매우 효과적임을 확인한다. 대부분의 Multi-time retrieval-augmented 방식이 single-time 보다 우수한 결과를 보이지만 그 간격은 다르다. Previous-sentence 을 사용하여 검색하는 개선은 비교적 작은데, 이는 2WikiMultihopQA의 다음 문장과 다른 entity 나 관계를 자주 설명하기 때문이라고 추측한다. 반면, Previous-window 접근 방식은 두 번째 절반을 생성하는 데 도움이 될 수 있는 정보를 검색하기 위해 문장의 첫 절반 부분을 쿼리로 사용할 수 있습니다. 모든 베이스라인 중에서 Query Decompoistion 인 Self-ask 가 가장 우수한 성능을 달성한다. 이는 in-context exexmplar 가 분해된 하위 질문(Prompt 4.1)으로 manually annotation 이 달려 있어 LM 이 미래 생성의 주제/의도와 일치하는 적절한 하위 질문을 생성하도록 안내되기 때문이다. FLARE은 이 베이스라인을 능가하며, manual exemplar annotation 이 미래를 고려한 효과적인 검색에 필요하지 않음을 나타낸다. $FLARE_{instruct}$와 Query decomposition 간의 차이는 크며, task-generic retreival instruction 과 exemplar 를 사용하여 LM 에게 검색 쿼리를 생성하는 방법을 가르치는 것이 어려움을 나타낸다.

다른 데이터셋에 대한 모든 metric 들은  Table 2에 있다. 다시 한 번, FLARE은 모든 지표에 대해 베이스라인을 능가합니다. Previous-window 을 사용한 검색은 ASQA 에서 single-time retrieval 보다 성능이 낮습니다. 이는 previous-window 가 사용자의 미래 의도를 정확하게 반영하지 못하기 때문이라고 가설을 세우고 있다. 저자들은 생성의 Factuality 를 평가하는 데 중점을 둠으로써 EM, Disambig-F1, UniEval과 같이 사실적인 콘텐츠를 강조하는 지표가 모든 토큰을 기반으로 계산된 지표(ROUGE-L 등)보다 더 신뢰성이 있다고 여긴다.

# Ablation study
<span style='color:green;font-weight:bold'> Importance of forward-looking retrieval</span><br>
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/19918d41-05a5-42b9-bdbe-61c885b7e9fc)

저자는 forward-looking 검색이 past-context-based retrieval 보다 실제로 강력한지 여부를 먼저 확인한다. 2WikiMultihopQA 및 ASQA-hint 데이터셋에서 ablation study 를 수행하여 previous 문장 대신 next 문장을 사용한 검색을 비교한다. 이때 두 가지 방법은 검색에 사용되는 쿼리를 제외하고 동일하다. 구체적으로, 두 가지 방법은 각 문장을 검색하고 검색에 전체 문장을 직접 사용한다. (마스킹 또는 질문 생성 없이). 위의 Table 3 에서 볼 수 있듯이, 두 데이터셋 모두에서 다음 문장을 사용한 검색이 이전 문장을 사용한 것보다 훨씬 더 나은 결과를 나타낸다.

<span style='color:green;font-weight:bold'> Importance of active retrieval</span><br>
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/7903808d-40fc-4f31-a794-2297d68068f0)

Threshold θ 와 performance 의 관계를 조사한다. 아무 것도 검색하지 않는 것(θ=0)에서 모든 문장을 검색하는 것(θ=1)으로 FLARE 방법을 변경하기 위해 검색을 트리거할 때 사용되는 θ를 0에서 1로 조정했다. 모든 thershold 에 대해 검색이 trigger 되는 단계/문장의 percentage 을 계산하고 검색의 percentage 을 기반으로 성능을 표시한다. Figure 5에서 볼 수 있듯이, 2WikiMultihopQA에서는 검색 비율이 60%를 넘어가면 성능이 안정화되며, LM d이 확신을 가질 때 검색이 필요하지 않음을 나타낸다. StrategyQA에서는 검색 비율이 50%를 넘어가면 성능이 하락하며, 고신뢰 문장을 검색에 사용하면 noise 가 끼고 원래 생성 프로세스를 방해할 수 있음을 시사한다. Task/Dataset에 따라 평균적으로 문장의 40%-60%에 대한 검색 트리거가 성능을 향상시키는데 일반적으로 좋은 결과를 나타낸다.

<span style='color:green;font-weight:bold'> Effectiveness of different query formulation methods</span><br>
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/10b1ab0f-33a9-4d9d-9823-7242e23ea06b)

마지막으로, Masking 을 통한 implicit query formulation 과 question generation 을 통한 explicit query formulation 에 대해 연구한다. Table 4에서 다른 threshold β로 FLARE의 성능을 비교한다. 완전한 문장을 직접 검색하는 것(β = 0)은 낮은 확률로 마스킹된 토큰보다 성능이 나쁘며, 낮은 신뢰도의 error token 이 retriver 를 distraction 할 수 있다는 것을 검증한다. 또한 implicit 및 explicit query formulation 방법을 Table 5 에서 비교한다. 두 방법의 성능은 유사하며, 두 방법 모두 정보 요구를 효과적으로 반영할 수 있다는 것을 나타낸다.

# Conclusion
To aid long-form generation with retrieval augmentation, we propose an active retrieval augmented generation framework that decides when and what to retrieve during generation. We implement this framework with forward-looking active retrieval that iteratively uses the upcoming sentence to retrieve relevant information if it contains lowconfidence tokens and regenerates the next sentence. Experimental results on 4 tasks/datasets demonstrate the effectiveness of our methods. Future directions include better alternatives for active retrieval and developing LM architectures for efficient active retrieval augmentation.
