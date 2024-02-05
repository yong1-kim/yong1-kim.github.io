---
layout: post
title:  "[ICML2022] HyperPrompt: Prompt-based Task-Conditioning of Transformers"
date:   2024-02-05 09:08:00 +0900
use_math: true
categories: [LLM, PLM]
---

[[pdf]](https://proceedings.mlr.press/v162/he22f/he22f.pdf)


**Yun He <sup>1*</sup>, Huaixiu Steven Zheng <sup>1*</sup>, Yi Tay <sup>2</sup>, Jai Gupta<sup>2</sup>,  Yu Du <sup>2</sup>, Vamsi Aribandi <sup>2</sup>, Zhe Zhao <sup>2</sup>, YaGuang Li<sup>2</sup>, Zhao Chen<sup>3</sup>, Donald Metzler<sup>2</sup>, Heng-Tze Cheng<sup>2</sup>, Ed H. Chi<sup>2</sup>**
<br><sup>1</sup> Texas A&M University <sup>2</sup> Google Research <sup>3</sup> Waymo LLC. &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/096235d2-fe66-4059-b1c8-91e6726c03e1)

## Abstract
- (<span style='color:green;font-weight:bold'> Hyperprompt </span>) 이 논문에서는 Transformers 속의 self-attention 에 prompt-based task-conditioning architecture 인 **Hyperprompt** 를 제안한다.
- (**Global memory**) HyperNetwork 를 활용하는 hyperprompt 는 task 간의 정보 교환을 해주는 역할에 더불어, task 의 global memory 의 역할을 한다는 것을 보인다.
- (**Efficiency**) 단지 0.14% 의 추가적인 param 만으로 T5 등의 multi-task learning baseline 과 비교하여 competetive 한 성능을 보인다.

## 1. Introduction
<span style='color:green;font-weight:bold'> ▶HyperPrompt </span>
<br>
Soft Learnable memory token 으로 LLM 을 condition 하는 prompt tuning 이 주목을 받고 있다.
Pretrained model 은 frozen 한 채 빠르고 가볍게 학습할 수 있다는 장점을 갖고 있다.

이 논문에서는 Multi-task learning 을 위한 새로운 Prompt-tuning 방법론인 HyperPrompt 를 제안한다.
<span style='background-color: #dcffe4'> HyperPrompt 는 task-conditioned hyper-prompt 를 도입하여, prompt 를 통해 task-specific information 을 모델이 condition 할 수 있게 한다. </span>

<span style='color:green;font-weight:bold'> ▶HyperNetwork </span>
<br>
저자들은 이 hyperprompt 를 위하여, HyperNetwork 를 도입한다.
이 HyperNetwork 가 task-aware and layer-aware prompt 를 generation 한다.
보통 기존의 multi-task learning 방법들은 task 수에 linear 하게 param 이 증가하기 마련인데, <span style='background-color: #dcffe4'> HyperNetwork 를 활용하면 아주 적은 양의 추가적인 param 만으로 기존의 방법들과 competitive 한 성능을 보일 수 있어 효율적이다. </span>
그리고 이들은 **prompt generator 의 개념은 HyperNetwork 가 처음**이라고 주장한다.


<span style='color:green;font-weight:bold'> ▶Training whole network including LM </span>
<br>

이들은 기존의 prompt 학습 방식이나 adapter 와 같은 개념과 다르게 LM 을 포함한 network 전체를 학습시키는 것이 중요하다고 한다. 그 이유로는 (1) 기존의 prompt-tuning 은 11B 이상의 LLM 에 대해서만 잘 적용이 되며, (2) adaptive param 만 학습한다고 해서 inference 에서 딱히 이득이 없다고 한다. 따라서, Network 를 전체 학습하여 성능을 높이는 것이 더 낫다고 판단한다.

## 2. Methods

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/6610e51c-79c5-4dc0-9917-3a760e8f882e)


HyperPrompt 에는 세 개의 변형 : **HyperPrompt-Share, HyperPrompt-Sep** 그리고 **HyperPrompt-Global**  이 있다.

가장 중요한 기본 개념은 <span style='background-color: #dcffe4'> (1) task-condtioning 을 self-attention 이 넣는 것, 그리고 (2)Prompt generation 을 위해 HyperNetwork 를 활용하는 것</span>이다.

# 2.1. Prompt-based Task-Conditioned Transformer

기존의 adtaper-based 방법들은 adapter(dense-relu-dense network) 를 Transformer block 의 FFN 직후에 집어넣는 방법들이었다.
<span style='background-color: #dcffe4'> Hyperprompt 에서는 대신 각각의 layer 에 task-conditioned trainable vector 를 key 와 value 앞에 prepend 한다. </span>
Netowrk 앞에 learnable prompt 를 prepend 하는 것은 이미 여러 연구가 존재하지만, multi-task learning 을 위하여 이 아이디어를 적용하는 것은 처음이라고 주장한다.

이를 위해 저자들은 <span style='color:green;font-weight:bold'> HyperPrompt </span> 방법을 제안한다.
위의 그림의 (a) 에서 보듯이 Key 와 Value 앞에 HyperPrompt 를 prepend 한다.
이후 기존 Transformer 방식처럼 Self-Attention 을 진행한다.
이는 장점이 있는데, Hyperprompt 가 attention feature map 형성에 관여한다는 점이 task-speific memory 로써 역할을 할 수 있다.

# 2.2. HyperPrompt
m-th layer 의 hyperprompt 를 어떻게 생성할 것인가에 대하여, 나이브하게 layer 마다 T(# of task)를 만든 다음 random init 하면 되지만, 이 경우 O(T X M) (M: # of layer) 로 비효율적이라고 한다.

이들은 우선, task 별로 global prompt 를 만든 다음, 이 global prompt 를 각 layer block 으로 projection 하여 M 개를 얻는 방법을 택한다.

<span style='color:green;font-weight:bold'> (1) Global Prompts </span>
<br>
첫 번째로, Task 개수 T 만큼의 global prompt 를 init 한다.

<span style='color:green;font-weight:bold'> (2) Local HyperNetworks </span>
<br>
각각의 Transfomer layer block 에서, 두 local HyperNetwork 가 global prompt 를 입력으로 받아, key local prompt 와 value local prompt 를 생성한다.
HyperNetwork 는 위의 figure (b) 에서 보듯이, down-projection 을 포함한 bottleneck architecture 를 활용한다.

<span style='color:green;font-weight:bold'> (3-1) HyperPrompt-Share </span>
<br>
앞서 말한 key, value local prompt 생성을 위한 hypernetwork 를 Task 마다 다르게 하지 않고, 모두 share 하는 setting 이다. 이 경우, parameter 는 많이 saving (1/T 로) 할 수 있겠지만, 실험 결과 모델 capacity 가 줄어든다고 한다.

<span style='color:green;font-weight:bold'> (3-2) HyperPrompt-Sep </span>
<br>
따라서 그 반대로, 각각의 task 마다 own local HyperNetwork 를 갖게하는 HyperPrompt-Sep 방법의 성능이 더 좋다고 한다. 

# 2.3. HyperPrompt-Global
그리고 다시 이 task-specific and layer-specific HyperNetwork 를 효율적으로 생성하기 위하여, Figure (c) 와 같이, global HyperNetwork 인 HyperPromt-Global 을 도입한다.
이는 Lyaer-Aware Task embedding 을 입력으로 받아, GLobal HyperNetwork 를 통해, 각 Layer 별 Hypernetwork 를 생성한다.

## 3. Experiments
# 3.1. Experimental Setup
- Dataset : GLUE, SUPERGLUE
- Transformers : T5-Base to T5-XXL
- Baselines : vanilla T5, vanilla Adapter, HyperFormer++ (adapter-based MTL model), Prompt-Tuning

# 3.2. Key Results

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/d6793f96-8e2e-4274-af5e-722666718a6e)

<span style='background-color: #dcffe4'> (1) Prompt-tuning 은 11B 모델에서만 잘 작동된다. </span>

<span style='background-color: #dcffe4'> (2) HyperPrompt 가 모든 모델 사이즈 전반에 걸쳐 좋은 성능을 보인다. </span>

# 3.3. Tuning all vs Task-Conditioned Params


기존의 연구에서, LM 을 전부 tuning 하는 것보다 prompt 만 tuning 하는 것이 더 좋다는 연구 결과가 있었지만, 그 연구는 GLUE benchmark 에 대해서 작은 모델인 T5 base, T5 small model 에 대해서만 측정했다고 한다.

이 실험에서는, full model 과 task-conditioned param 만 학습하는 것을 비교실험한다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/8e63ea46-5a50-4e40-91cb-26a81a7e1e54)

이 실험에서 보듯이 HyperPrompt 를 활용하는 경우 Full Model 을 tuning 하는 것이 훨씬 좋은 성능을 보인다. 

# 3.4. Computational Efficiency

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/6578a602-d511-41dd-acf1-1af051e11089)

<span style='background-color: #dcffe4'> HyperPrompt 는 FFN 을 사용하지 않고, self-attention 에 버무려지기 떄문에, 더 적은 #Ops 를 가진다. </span>
또 추가적으로, Training Time 역시 효과적이다.

# 3.5. Ablation Study

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/1812365f-39ac-482a-b66a-8d1922a26471)

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/d55e59be-1c34-4135-908f-357f1d212bf7)

위의 표는 T5 base, 아래 표는 T5 large 에서의 실험 결과이다.

<span style='color:green;font-weight:bold'> (1) HyperPrompt-Global vs Prompt-Tuning. </span>
<br>
Prompt-Tuning 은 single task finetuning 과정이고, LM 전체를 tuning 하지 않기 떄문에 Fair 비교를 위해 Task 별 prompt 를 추가하고 LM 전체를 tuning 하여 비교한다.
실험 결과, GLUE 와 SUPERGLUE 에서 모두 더 좋은 성능을 보인다.

<span style='color:green;font-weight:bold'> (2) HyperPrompt-Global vs HyperFormer++. </span>
<br>
Adapter-based 방법인 HyperFormer++ 와의 비교에서도 우위의 성능을 보인다.

<span style='color:green;font-weight:bold'> (3) HyperPrompt-Global vs MTL. </span>
<br>
Multi Task Learning 을 통해 task 여러 개를 다 학습한 모델과 비교했을 때, 아주 적은 양의 Additional Param (1.02배)로 성능향상을 이끌어낸다.

<span style='color:green;font-weight:bold'> (4) HyperPrompt-Global vs HyperPrompt-Share/Sep. </span>
<br>
놀랍게도 HyperPrompt-Share 모델이 Sep 보다 SUPERGLUE 에서는 더 성능이 좋다.
그리고 projection network 를 생성하는 global HyperNetwork 를 쓰는 HyperPrompt-Global 이 모든 경우에서 가장 좋은 성능을 보인다.

## 4. Conclusion
```
We propose a novel architecture for prompt-based taskconditioning of self-attention in Transformers. The hyperprompts are generated by a HyperNetwork to enable flexible information sharing among tasks while remain efficient in parameters and computation. HyperPrompt allows the network to learn task-specific feature maps where the hyper-prompts serve as task global memories, encouraging a more diverse distribution of attention. Extensive experiments show that HyperPrompt can achieve superior performances over strong T5 multi-task learning baselines and parameter-efficient models including Prompt-Tuning and HyperFormer++ on GLUE and SuperGLUE benchmarks.
```
