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
<br>
<span style='color:red'> The quick brown </span>  fox jumps over the lazy dog.

<span style='color:red'>for(i= </span> 0;i<n;i++) sum+=array[i]

위의 두 문장은 인간이 만든건지 머신이 만든건지 구분이 힘들다. 왜냐하면 이들은 낮은 entropy 를 갖고 있기 때문에, first few toekn 은 strongly determined 되기 때문이다.
따라서 인위적으로 watermark 를 이러한 low entropy 문장에 집어넣는 것은 오히려 perplexity 를 크게 높이는 결과를 가져와 quality 를 떨어뜨린다.

# A simple proof of concept
![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/9982ebac-7f87-4710-9b49-1a2e947ebf9e)

첫 번째로는 <span style='background-color: #dcffe4'> simple "hard" red list wateramrk </span> 를 통한 기법이다. 이는 분석하기 쉽고 찾아내기도 쉬우며 remove 하기는 어렵다. 
그러나 이 방식은 low entropy 문장에 대해서 poor generation quality 를 초래하는 cost 를 수반한다.

이 방식은 위의 알고리즘에 볼 수 있다. 이 방식은 t-token $s^t$ 에서 등장할 수 없게 하는 토큰들의 집합 pseudo-random red list 를 생성한다. 이 red list 는 $s^{(t-1)}$ 에서 seeded 되기 때문에 entire sequence 에 대한 접근 없이 reproduce 가능하다.

<span style='color:green;font-weight:bold'> Detecting the watermark. </span><br>

워터마크가 있는 텍스트를 생성하는 데는 언어 모델에 대한 액세스가 필요하지만, 워터마크를 감지하는 데는 그러한 액세스가 필요하지 않는다. 해시 함수와 난수 생성기에 대한 지식을 가진 제3자 (third party) 는 각 토큰에 대한 빨간색 목록을 다시 생성하고 빨간색 목록 규칙이 얼마나 자주 위배되는지 계산할 수 있다. 

다시 말해 아래의 *null hypothesis* (귀무가설) 에 대한 test 로 watermark 를 detect 할 수 있다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/f40ae537-5935-4b6c-be3a-29b5b444ed7a)

왜냐하면, red list 가 매번 무작위로 선택되기 때문에 natural writer 는 자연스럽게 자신의 토큰 중 절반 정도에 대해 red list 를 위반할 것으로 예상되며, 반면 워터마크가 있는 모델은 위반을 생성하지 않을 것으로 예상되기 때문이다. 
Red list 규칙을 위반하지 않고 $T$ 개의 토큰을 생성하는 확률은 $1/2^T$ 이다.
이는 심지어 몇 마디로 이루어진 짧은 텍스트 조각에 대해서도 거의 없는 확률을 의미한다.

<span style='background-color: #dcffe4'> 귀무가설을 검증하기 위한 더 견고한(Robust) 감지 방법은 one proportion z-test를 사용하는 것이다. </span> 
만약, 귀무가설이 참이라면, green list token 의 수 $s_{G}$ 는 $T/2$ 의 value 와 variance $T/4$ 일 것이다. 따라서 z-statistics 는 아래와 같다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/6d40dec7-4f83-46f1-9ff2-c43344b3bc11)

$z$ 가 특정 threshold 이상이면 이 귀무가설을 reject 하고 watermark 가 존재한다고 주장할 수 있다. 만약 $z$ > 4일 경우 귀무가설을 기각하기로 선택한다고 가정하면, 이 경우 false positive 의 확률은 $3 × 10^{(-5)}$ 이다. 이는 $z$ > 4에 해당하는 one-sided p-값이다. 동시에 $T$ 값이 16 이상인 경우 ($|s|_G=T$에서 $z$ = 4를 만드는 최소값) 어떠한 워터마크가 있는 시퀀스도 감지할 것이다.

<span style='color:green;font-weight:bold'> How hard is it to remove the watermark?  </span><br>
**One proportion $z$-test 를 사용하면 워터마크를 제거하기가 어려워진다.** 
길이가 1000 인 워터마크가 있는 시퀀스에 대해, 만일 적대적인 사용자 (Adverary) 가 시퀀스에서 200 개의 토큰을 수정하여 Red list 단어를 추가하고 워터마크를 지우려고 한다면, 위치 t의 수정된 토큰은 위치 t에서 Red list 규칙을 위반할 것이다.
게다가 $s^t$의 값은 토큰 $s^{t+1}$ 에 대한 Red list 를 결정하기 때문에, $s^t$ 를 최대한 적대적으로 선택하면 $s^{t+1}$ 이 Red list 규칙을 위반하게 만들 수 있다. 따라서 200 개의 토큰 뒤집기로 인해 최대 400번의 빨간색 목록 규칙 위반이 발생할 수 있다.

공격자에게는 불행하게도, 이 최대한 적대적인 시퀀스에서조차 600개의 남은 녹색 목록 토큰은 여전히 z-통계량 2(600 - 1000/2)/√1000 로 계산되며 이는 약 6.3이며, p-값은 약 10^(-10) 정도이다. 
이는 워터마크를 극도로 높은 신뢰도로 쉽게 감지할 수 있게 해준다. 일반적으로 긴 시퀀스의 워터마크를 제거하려면 대략 토큰의 1/4 이상을 수정해야 한다.

위의 분석은 공격자가 워터마크에 대한 완전한 지식을 가지고 있으며 각 선택된 토큰이 최대한 적대적인 경우를 가정한다 (이는 품질에 부정적인 영향을 미칠 가능성이 높다).
실제로, 워터마크 알고리즘을 알지 못하는 경우, 각 뒤집힌 토큰이 빨간색 목록에 속할 확률은 50% 뿐이며, 인접한 토큰도 동일합니다. 
이 경우 공격자는 200개의 토큰을 수정하여 (기대값 상으로) 200개의 빨간색 목록 단어를 생성한다. 

정리하면, 인간이 직접 생성한 문장은 약 50% 정도가 red list 에 속하는 단어들일 것이다. 
공격자가 워터마크를 제거하기 위해, Red list 를 알고 있다고 가정해도, 문장의 20% 정도를 뒤집는 것으로는 워터마크를 제거하기 힘들며, 최소한 25% (1/4) 정도 token 을 뒤집어야만 귀무가설을 reject 하고 인간이 생성한 문장이라고 주장할 수 있을 것이다. 그만큼 'hard red list' 규칙으로 생성된 watermark (green 의 연속)은 제거하기가 힘들다.  

<span style='color:green;font-weight:bold'> Drawbacks of the hard red list rule.  </span><br>

하지만, "Hard red list rule"은 낮은 엔트로피 시퀀스를 너무나도 간단한 방법으로 처리하여 문제가 된다. 이 규칙은 언어 모델이 low entropy 시퀀스를 생성하는 것을 방지하기 때문에 문제가 발생한다. 
예를 들어, "Barack" 토큰은 많은 텍스트 데이터셋에서 거의 결정적으로 "Obama"가 뒤를 따를 것이지만, "Obama"가 빨간색 목록에 포함되어 있을 수 있다.

<span style='background-color: #dcffe4'> 따라서, "Soft" watermarking 규칙을 사용하는 것이 더 나은 behavior 이다. </span> 이 규칙은 감지하기 어려운 high-entropy 텍스트에서만 활성화된다. 
충분히 높은 total entropy 속에 low entropy sequence 가 쌓여 있는 상황에서도, 해당 구문은 여전히 워터마크 감지기를 쉽게 작동시킬 수 있어서 1.2절에서 설명한 문제를 해결할 수 있다. 
더 나아가, Beam search decoder 와 워터마크를 결합할 수 있다. "Irons-in" 하는 빔 서치 디코더를 사용하면 가능성 있는 토큰 시퀀스의 가설 공간을 검색하여 녹색 목록 토큰이 높은 밀도로 나타나는 후보 시퀀스를 찾을 수 있으며, 이는 최소한의 혼란 비용으로 높은 강도의 워터마크를 생성한다.

# A more sophisticated watermark
이제 "Soft" Watermark 대해 알아보자. 
**짧게 정리하면 Red list 에 있으면 절대 뽑히지 않는 "Hard" Red list rule 과 다르게, Red list 에 속해 있더라도 뽑힐 수 있는 확률을 갖는다.**
이는 많은 좋은 선택지가 있는 고 엔트로피 토큰에 대해 green list 의 사용을 촉진하면서 거의 결정적인(deterministic 한) 낮은 엔트로피 토큰의 선택에는 거의 영향을 미치지 않는다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/ac2734dd-d517-4d0f-9cb0-442e2eae117d)

LLM 은 위와 같이 마지막 layer 의 logit 값의 softmax 를 통해 vocab 의 확률 벡터 p 를 결정한다. "Soft" watermark 는 여기에 hardness parameter $\delta$ 를 추가한다. 그리고 0.5 의 확률 대신 green list size $\gamma$ 를 도입한다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/58dfacd7-dcd6-4564-875e-abeb7a60dfd2)

"Soft" red list rule 은 워터마크가 품질에 거의 영향을 미치지 않을 상황에서 워터마크를 강제하면서, 엔트로피가 낮은 경우에는 거의 워터마크 규칙을 무시한다.
다시 말해, $p(t)k ≈ 1$을 가진 highly-likely word 는 다른 후보보다 훨씬 큰 로짓을 갖고 있으며, 이는 **Red list 에 포함되어 있더라도 가장 큰 값을 유지한다.**
그러나 엔트로피가 높은 경우에는 선택할 수 있는 많은 유사한 logit 들이 있으며, 이 때 δ 규칙은 샘플링 분포에 큰 영향을 미치며 결과를 녹색 목록 쪽으로 강하게 편향시킨다.

<span style='color:green;font-weight:bold'> Detecting the soft watermark. </span><br>
"Soft" Watermark 를 감지하는 과정은 기존 "Hard" Watermark 탐지와 동일하다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/196c4f19-e90b-464b-87bc-1571778a58ad)

임의의 $\gamma$ 에 대해서 $z$ value 는 위와 같다. 
$z>4$ 인 경우를 다시 한 번 생각하면, 여전히 False-positive 는 $3 × 10^{(-5)}$이다.
"Hard" watermark의 경우, 텍스트의 특성과 관계없이 16개 이상의 토큰으로 이루어진 어떠한 워터마크가 있는 시퀀스라도 감지할 수 있었지만, "soft" watermark의 경우, watermark 텍스트를 감지하는 능력은 시퀀스의 엔트로피에 따라 달라진다.
높은 엔트로피 시퀀스는 상대적으로 적은 토큰으로 감지되지만, 낮은 엔트로피 시퀀스는 감지를 위해 더 많은 토큰이 필요하다. 

# Analysis of the soft watermark
이 섹션에서는 'soft' watermark 에 대한 보다 면밀한 분석을 진행한다.
실제 sampling method 와 다르게, red list 는 uniform 하게 sample 된다고 가정한다. 실제로는 previous token 을 seed 로 하여 random number generator 에 의해 sample 된다.

분석을 위해 "Spike" 라고 하는 새로운 entropy 개념을 정의한다. discrete probability mass $p$ 와 scalar $z$ 에 대해 'spike' 는 아래와 같다.

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/f1ac1975-60b8-480f-82f6-c8f4d29af7cb)

이 것은 기존의 Shannon entropy 와 유사하게, single location 에 mass $p$ 가 몰려있을 때 가장 적은 $1/{1+z}$ 값을 가지며, uniformly distritubted 되었을 때, 가장 큰 값인 $N/{N+z}$ 를 가진다.

