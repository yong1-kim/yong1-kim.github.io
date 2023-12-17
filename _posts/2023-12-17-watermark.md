---
layout: post
title:  "[ICML2023] A Watermark for Large Language Models"
date:   2023-12-17 14:05:00 +0900
use_math: true
categories: [Transformer, LLM]
---

[[pdf]](https://proceedings.mlr.press/v202/kirchenbauer23a/kirchenbauer23a.pdf)
[[github]](https://github.com/jwkirchenbauer/lm-watermarking)

**John Kirchenbauer <sup>1*</sup>, Jonas Geiping <sup>1*</sup>, Yuxin Wen <sup>1</sup>, Jonathan Katz <sup>1</sup>, Ian Miers <sup>1</sup>, Tom Goldstein <sup>1</sup>**
<br><sup>1</sup> University of Maryland. Correspondence to: John Kirchenbauer <jkirchen@umd.edu>.
 &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/17fdea52-8a0b-45cb-a818-3a38e9c7eac7)

# Abstract
- (Motivation) LLM 의 potential harm 은 모델 출력물에 <span style='color:green;font-weight:bold'> watermarking </span> 을 적용함으로써 완화될 수 있다. 인간의 눈에는 보이지 않지만 알고리즘적으로는 짧은 토큰 범위에서 감지할 수 있는 신호를 생성된 텍스트에 포함시키는 것이다. 
- (**Watermark**) 텍스트 품질에 미미한 영향을 미치도록 Watermarking 을 내장할 수 있으며, 언어 모델 API나 매개변수에 접근하지 않고도 효율적인 오픈 소스 알고리즘을 사용하여 감지될 수 있다.
- (Method) Word token 이 생성되기 전에 랜덤화 된  <span style='background-color: #dcffe4'> green token </span> 을 생성하고, sampling 과정에서 이 <span style='background-color: #dcffe4'> green token </span> 을 사용한다.
- (Experiment) Interpretable p-value test 와 정보이론 framework 을 통해 watermark 의 sensitivity 를 검증하고, multi-billion parameter 를 갖는 OPT (Open Pretrained Transformer)
family 에 대해 실험을 검증하였다.

# Introduction
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/832d0665-cfea-433e-9c86-4151f415c64c)
Large Language Model (LLM) 의 성능이 폭발적으로 증가함에 따라, fake-news 나 fake web contents 등의 생성을 통한 정치 공작, AI system 을 활용한 academic writing 에서의 cheating 문제 등의 사회 문제로 자리잡고 있다.
게다가, LLM 을 통한 synthetic web data 의 증가(proliferation)하고 있는데, 이들은 human-annotated data 에 비해 품질이 매우 저조하기 때문에 model training 직전에 detected 되고 제거되어야 한다.
<span style='background-color: #dcffe4'> 이러한 이유로, machine-generated text 에 대한 detection 과 audit 은 필수적이다. </span>

이 논문에서는 <span style='color:green;font-weight:bold'> watermarking </span> 기법 제안한다. watermark 는 인간은 알아차리기 힘들지만, text 를 synthetic 하다고 identifiable 하게 하는 특정한 hidden pattern 이다. 저자들은 25 token 이내 정도의 short span 으로 synthetic text 를 detect 할 수 있는 efficient watermark 를 제시하고, 이는 human-generated text 를 detect 하는 false-positive 가 통계학적으로 불가능한 획기적인 방법이다.

논문에서 제안하는 Watermark 의 주요 특징을 정리하면 아래와 같다.
(1) Language model API 에 대한 접근 혹은 model param 의 사전 지식과 완전히 무관하게 적용이 되는 알고리즘이다.
(2) Re-training (e.g. finetuning) 이 없이도 적용 가능하다.
(3) Watermark 는 generated text 의 일부만을 활용하기 때문에, 큰 문서의 일부만 생성이 되는 경우에도 detect 가능하다.
(4) 워터마크는 생성된 토큰의 상당 부분을 수정하지 않고는 제거할 수 없다.
(5) 워터마크가 감지되었을 확신의 엄격한 통계적 측정치를 계산할 수 있다.

위의 Figure 가 watermark 를 감지하는 방법이다. 맨 아래의 watermark 가 적용된 text 의 경우, 만약 인간이 쓴다면 9 개의 green token 이 예상되지만, 28개의 green token 이 감지된다. 통계학적으로 이런 일이 일어날 확률은 6e-14 로, 강력하게 machine 이 만든 텍스트임을 확신할 수 있다. 

<span style='color:green;font-weight:bold'> A caveat: The difficulty of watermarking low-entropy sequences </span>
<br>
아래의 두 문장을 보자.
<span style='color:red'> The quick brown </span>  fox jumps over the lazy dog.

<span style='color:red'>for(i= </span> 0;i<n;i++) sum+=array[i]

위의 두 문장은 인간이 만든건지 머신이 만든건지 구분이 힘들다. 왜냐하면 이들은 낮은 entropy 를 갖고 있기 때문에, first few toekn 은 strongly determined 되기 때문이다.
따라서 인위적으로 watermark 를 이러한 low entropy 문장에 집어넣는 것은 오히려 perplexity 를 크게 높이는 결과를 가져와 quality 를 떨어뜨린다.

# A simple proof of concept
![Uploading image.png…]()





