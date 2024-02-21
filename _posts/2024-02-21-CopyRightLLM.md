---
layout: post
title:  "[EMNLP 2023] Copyright Violations and Large Language Models"
date:   2024-02-21 20:27:00 +0900
use_math: true
categories: [Retrieval, LLM, PLM]
---

[[pdf]](https://aclanthology.org/2023.emnlp-main.968.pdf) &emsp;
[[github]](https://github.com/coastalcph/CopyrightLLMs)

**Antonia Karamolegkou<sup>1*</sup>, Jiaang Li<sup>1*</sup>, Li Zhou<sup>12</sup>, Anders Søgaard<sup>1</sup>**
<br><sup>1</sup> Department of Computer Science, University of Copenhagen <sup>2</sup> University of Electronic Science and Technology of China
 &emsp;

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/85b6b96d-a378-4b28-852b-c3416bc088a5)

## Abstract
- (**verbatim memorization**) 언어 모델은 훈련 중 본 텍스트의 전체 chunk 를 포함하여 사실 이상의 것을 기억할 수 있다.
- (**Copyrighted text and LLMs**) 이 연구는 LLM 의 copyrighted text 에 대한 침해 문제를 정확한 복제 기억의 관점에서 탐구하며, copyrighted text의 redistribution 에 초점을 맞춘다.

## Introduction

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/7f4f8639-d0d8-4870-be1f-97ef4ad39a8c)

<span style='color:green;font-weight:bold'> ▶ Verbatim memorization </span>
<br>
당신이 "오만과 편견" 에 대해서 이야기를 하거나, 관련한 글을 작성할 때, 완전히 같은 내용을 작성하여 저작권(copyright)을 침해하는 일은 벌어지지 않을 것이다. 
하지만 LLM 에게 시켜보면 어떨까?

ChatGPT 에게 성경의 첫 50 줄을 출력하게 시키면, training data 을 memorize 하여 완벽히 구절을 그대로 읊는 것을 볼 수 있다. 이렇게 LLM 이 training data 를 memorization 하는 것은 이제 어제오늘의 일이 아니다.

기존에도 Copyrighted book 에 대해 lagnuge model 의 memorization 을 probing 하는 연구는 있었다.([[1]](https://arxiv.org/abs/2305.00118))
그러나 이 연구는 <span style='background-color: #ffdce0'> cloze-style 에 국한되었고, ad verbatim (말 그대로의) memorization setting 은 아니었다. </span>
이 연구에서는 copyrighted text 의 문장들을 **"말 그대로(verbatim)" 가져오는 것**에 대한 probing 을 다룬다. 
<span style='background-color: #dcffe4'> 과연 LLM 은 copyrighted text 에 대한 관련 법률을 지켜낼 수 있을까?</span>

이 연구에서는 best-seller book 과 LeetCode 에 대해 probing 실험을 진행해본다.
그 결과, copyrighted book 뿐 아니라 leetcode 등 저작권이 있는 글과 코드에 대해서, 저작권 침해의 복제가 일어날 수 있음을 확인한다.

## CopyRight Laws

저작권 법과 규약은 창작자들에게 그들의 창작물을 사용하고 배포할 수 있는 독점적 권리를 부여한다. 
단, 특정 예외가 있다 (예를 들어, 1952년 9월 6일의 세계 저작권 협약, 베른 협약, 미국 저작권법 §106, 디지털 단일 시장에서의 저작권 및 관련 권리에 대한 유럽 의회의 지침 (EU) 2019/790 및 지침 96/9/EC 및 2001/29/EC의 수정). 
미국 저작권법 §107에 따르면, 공정 사용은 저작권 침해로 간주되지 않는 예외로, 예를 들어 도서관이나 기록 보관소가 직접적 또는 간접적 상업적 이익을 목적으로 하지 않고 문학 작품을 배포하는 경우가 해당되지만, 이는 세 부분까지로 제한된다. 
<span style='background-color: #dcffe4'> 이는 Large Language Models 제공업체들이 유명한 문학작품의 구절을 인용하는 것이 공정한지 여부를 주장해야 함을 의미한다.  </span>

유럽 Context 에서는 인용이 저작권의 예외 및 제한 중 하나로 2001/29/EC 정보 사회 지침의 저작권 및 관련 권리 조항에 나열되어 있다. 
이 법안은 회원국이 비판 또는 검토와 같은 목적을 위한 인용을 저작권 법의 예외로 제공할 수 있도록 규정하고 있으며, 이는 공개적으로 합법적으로 이용 가능해진 작품이나 다른 주제에 관련된 경우, **출처와 저자명이 불가능하지 않은 한 표시되며**, 공정한 관행에 따라 특정 목적에 필요한 범위 내에서 사용되어야 힌다. 
전체 인용을 생성하는 언어 모델은 저작권 위반을 피하기 위한 좋은 실천일 수 있다. 
그러나, <span style='background-color: #dcffe4'> 300 단어 이상을 그대로 인용하는 경우 공정 사용에 반대하는 판결 </span> 을 내릴 수 있는 상황도 존재한다. 
따라서, LM 이 **작은 텍스트 조각을 단순 인용**으로 배포하고 인용을 제공하더라도 여전히 저작권 법을 위반할 수 있다. 

마지막으로, 저작권 위반을 방지할 수 있는 또 다른 예외는 일반적인 관행(Common Practice) 이다. 
예를 들어, 책 길이의 자료에 대해 일부는 300 단어가 일반적인 관행이라고 하지만, 다른 이들은 25단어에서 1000단어까지 다양하게 주장할 수 있다. 
**Chapter, magazines, journals, teaching material 에 대해서는 50 단어가 일반적**입니다. 
<span style='color:green;font-weight:bold'> 
저자들은 책과 교육 자료(LeetCode 문제 설명)에 관심이 있었기 때문에, 기준으로 50단어를 설정했다. </span>

## Experiments

앞서 말했듯, LLM 이 Copyrighted book 과 LeetCode 에 대해 저작권 침해 문제를 일으키는지 실험적으로 확인한다.
Open-source model 로는 prefix probing 을 이용하고, closed-source instruction-tuned model 에는 direct probing 을 이용한다.
이 때 prompt 는 "What is the first page of [TITLE]?" 이다.

- **Datasets** : 1930-2010 best-sellers (Table below), Leetcode - coding challenge problems

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/aba91fd8-5813-4338-97f4-d547bf4d2424)

- **Language Models**

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/60b695ea-6c0c-4e90-865e-b72ae9e48951)


## Results and Discussion

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/6ffa1dea-44d7-469f-81aa-fc1fa4e42b16)

<span style='color:green;font-weight:bold'> Do larger language models memorize more? </span>
<br>

[Figure2 Left]

- <span style='background-color: #dcffe4'> 더 큰 언어 모델이 미래에 기존 저작권을 점점 더 침해할 수 있다는 우려가 있다 </span>

- <span style='background-color: #dcffe4'> 60B 미만의 모델도 평균적으로 간단한 프롬프팅 전략을 사용하여 50단어 미만의 기억된 텍스트를 재현 </span>

- <span style='background-color: #dcffe4'> GPT-3.5 와 Claude (둘 다 closed-source) 는 저작권 침해 문제가 심각하다. </span>

<span style='color:green;font-weight:bold'> What works are memorized the most?  </span>
<br>

[Figure2 Right]

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/9b5f1e28-6c0a-4333-8820-e2b8c2d668a5)

<span style='color:green;font-weight:bold'> Popularity indicators. </span>
<br>

- <span style='background-color: #dcffe4'> GPT-3.5 LCS 실험에서, Reveiw 와 Eidtion 이 클 수록 LCS length 가 커지는 양의 상관관계가 있다</span>

- <span style='background-color: #dcffe4'> LeetCode 에서는 Ranking 이 낮을 수록, 더 흔한 코드라 LCS ratio 가 크다.</span>

![image](https://github.com/yong1-kim/yong1-kim.github.io/assets/42200027/8c1851a4-ef80-4fad-88e5-f485d5696f4a)

## Conclusion
```
Overall, this paper serves as a first exploration of verbatim memorization of literary works and educational material in large language models. It raises important questions around large language models and copyright laws. No legal conclusions should be drawn from our experiments, but we think we have provided methods and preliminary results that can help provide the empirical data to ground such discussions.
```
