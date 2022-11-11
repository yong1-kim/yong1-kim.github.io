---
layout: post
title:  "[ICML2022] What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?"
date:   2022-11-11 15:48:00 +0900
use_math: true
categories: [LLM, Transformer]
---
[[pdf]](https://proceedings.mlr.press/v162/wang22u/wang22u.pdf)  &emsp;
[[github]](https://github.com/bigscience-workshop/architecture-objective) <br>

**Thomas Wang<sup>* 1</sup>, Adam Roberts<sup>* 2</sup>, Daniel Hesslow<sup>3</sup>, Teven Le Scao<sup>1</sup>, Hyung Won Chung<sup>2</sup>, Iz Beltagy<sup>4</sup>, Julien Launay<sup>3 5</sup>, Colin Raffel<sup>1</sup>**
<br><sup>*</sup> Equal Contribution, <sup>1</sup> Hugging face, <sup>2</sup> Google, <sup>3</sup> LightOn, <sup>4</sup> Allen institute for AI, <sup>5</sup> LPENS, Ecole Normale Superieure.   &emsp; 

![image](https://user-images.githubusercontent.com/42200027/201280889-750c58c2-b68b-4b58-9f82-2feb39f72b08.png)

# Abstract
- (Motivation) Large Language Model (LLM) 이 zero-shot generalization 에 좋은 성능을 보이지만, 많은 state-of-the-art 모델들이 각기 다른 architecture 와 pre-training objective 를 통해 학습되는데, 이 요소들에 대한 체계적인 비교(systematic comparison) 이 적다. 
- (Solution) 이 논문에서는 여러 LLM 들의 modeling choice 와 zero-shot generalization 에 대한 영향력을 평가하는 large-scale evaluation 방법을 제안한다.
- (Experiment) (causal decoder-only / non-causal decoder-only / encoder-decoder) 세 구조 (architecture)와 (autoregressive, masked language modeling) 두 가지 pre-training objective, 그리고 with/without multitask finetuning 조합들을 실험을 진행한다.
- (Results) causal decoder-only + autoregressive 방법이 zero-shot 성능은 제일 좋았으나, non-causal + mlm + multi-task finetuning 방법이 실험 성능은 제일 좋았다.
 
# Introduction
* 위의 그림의 module 블록들의 색과 아래의 색을 연결

Large Language Model(LLMs) 들은 unstructured text data 에 pre-train 된 후, additional training or labeled data 없이 다양한 방면의 task 에서 좋은 성능을 보인다. 이러한 능력을 <span style='background-color: #a5e5f2'> zero-shot generalization </span> 이라고 한다. 현재 대부분의 LLM 들은 <span style='background-color: #f7efb0'> [Transformer](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) architecure </span>  기반으로 구성되어 있다. Original Transformer 는 encoder-decoder 구조로 되어있지만, 많은 최근 LLM 들은 <span style='background-color: #f7efb0'> causal decoder-only </span> 모델로, auto-regressive 방법으로 학습한다([[1]](https://arxiv.org/pdf/1801.10198.pdf), [[2]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), [[3]](https://ojs.aaai.org/index.php/AAAI/article/view/4182)) 그러나, [T5 model](https://arxiv.org/pdf/1910.10683.pdf) 에서는 <span style='background-color: #f7efb0'> Encoder-Decoder (ED) </span> model 을 통해 transfer learning 으로 decoder-only LLM 을 outperform 한다. 추가적으로, [UniLM](https://arxiv.org/pdf/1905.03197.pdf) 등의 <span style='background-color: #f7efb0'> Non-causal decoders-only </span> 구조는 attnetion mask 를 활용하여 decoder-only 와 encoder-decoder model 의 구조 사이의 gap 을 줄인다.

최근의 연구([[4]](https://arxiv.org/abs/2110.08207),[[5]](https://arxiv.org/abs/2201.06910),[[6]](https://arxiv.org/abs/2109.01652))에서는 prompted task 들의 ensemble 의 <span style='background-color: #dcffe4'> multitask finetuning stage </span> 을 통해 encoder-decoder model 과 causal decoder-only 모델에서 엄청난 성능향상을 이끌어내었다. 이에 따라, multitask fintuning 의 conjunction 과 architecture choice 의 조합에 따른 zero-shot generalization 성능에 대한 의문점이 제기된다.

Transformer 모델들은 다양한 <span style='background-color: #f7b6b0'> self-supervised training objective </span> 를 가질 수 있다. 보통, causal decoder-only LLM 들은 <span style='background-color: #f7b6b0'> full language modeling (FLM) </span> objective 로, encoder-decoder model 은 <span style='background-color: #f7b6b0'> masked language modeling (MLM) </span> objective 로 학습을 진행한다. MLM 에는 span corruption 등이 포함될 수 있다. 추가적인 downstream task finetuning 에 대해, MLM 의 효과에 대해서는 이미 많은 연구에서 검증이 되었다. 최근 가장 강력한 성능을 보이는 [T0 model](https://arxiv.org/pdf/2110.08207.pdf) 역시 MLM 을 사용하였다. 최근, [Lester et al.](https://arxiv.org/pdf/2104.08691.pdf) 은 <span style='background-color: #fabbeb'> adaptation stage </span> (extending pretraining but with a different objective) 를 소개한다. 이는 MLM 모델을 prompted text generation task 를 수행하는 것을 가능하게 하며, objective 사이의 gap 을 줄여준다.

이러한 결과들은 <span style='color:green;font-weight:bold'> which architecuter and which pre-training objective pair </span> 가 LLM 에 가장 강력한 (strongest) zero-shot generalization capability 를 부여하는지에 대한 의문점을 남긴다. 기존에 이러한 구조와 목적함수 조합에 대한 연구가 있었지만, zero-shot 성능에 관한 연구는 아니었고, 대부분 transfer learning 을 위한 연구([[7]](https://arxiv.org/abs/2102.11972), [[8]](https://jmlr.org/papers/v21/20-074.html)) 였다. 또, 추가적으로 최근 multitask finetuning 이 효과적이라는 것이 증명되면서, 어떠한 조합이 multitask finetuning 과 잘 맞을지에 대한 궁금증도 생긴다.

<span style='color:green;font-weight:bold'> Large-scale systematic study </span>
<br>

이 논문에서는 <span style='background-color: #f7efb0'> architecture  </span> 와 <span style='background-color: #f7b6b0'> pre-training objectives </span> 의 조합에 따른 <span style='background-color: #a5e5f2'> zero-shot generalization  </span> 의 성능에 대한 실험을 진행한다. 그림에서와 같이, causal decoder-only, noncasual decoder-only, encoder-decoder architecture 들과, full, prefix, mlm 의 여섯 가지 조합으로 실험을 진행한다. 추가적으로, with and without <span style='background-color: #dcffe4'> multitask finetuning </span> 역시 평가한다. 
실험은 large-scale 로 진행한다. 5 billion parameter (11 billions for encoder-decoder) on 168 billion token 으로 학습하고, multitask finetuning 은 13 billion token 에 수행한다. Evaluation set 으로는 <span style='background-color: #a5e5f2'> [T0-Eval](https://arxiv.org/pdf/2110.08207.pdf) </span> 과 <span style='background-color: #a5e5f2'> Eleuther AI LM Harness (EAI-Eval) </span> 을 활용하고, 이들은 다양한 prompt 들의 30 개의 downstream task 를 갖고 있다. 

<span style='color:green;font-weight:bold'> Multitask finetuning impacts architecture and objective choice </span>
<br> 

저자들은 FLM objective 로 학습된 causal-decoder model 이 (GPT-3 와 유사) pre-training 직후 바로 zero-shot 을 잴 때 좋은 성능을 보이는 것을 발견했다. 그러나, multitask finetuning 을 진행한 이후에는 오히려 MLM 으로 학습한 모델이 더 좋은 결과를 보였고, FLM 으로 학습한 causal decoder-only 모델은 좋지 않았다.

<span style='color:green;font-weight:bold'> Bridging across architectures and objectives with adaptation </span>
<br>

여러 조합에 대한 adaptation 으로 두 가지를 고려한다.
첫 번째는 <span style='background-color: #fabbeb'> full language modeling adaptation</span> 으로 MLM-trained non-causal decoder model 을 FLM + causal decoder 로 변환한다. 이렇게 할 경우, FLM task 에서 1.6 배 빠르게 수렴을 한다. 두 번째는, <span style='background-color: #fabbeb'> non-causal MLM adaptation </span> 으로, FLM + causal decoder 를 MLM + non-causal decoder 로 바꾼다. 이렇게 바꾼 경우, MLM task 에 대해 3.3 배 빠르게 수렴한다. 이러한 adaptation 방법은 new version of model suited for multitask finetuning 을 생산하고, benchmark 에서 두 번째로 좋은 성능의 결과를 보인다.

