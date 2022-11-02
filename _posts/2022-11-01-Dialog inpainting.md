---
layout: post
title:  "[ICML2022] Dialog Inpainting: Turning Documents into Dialogs"
date:   2022-11-01 10:01:00 +0900
categories: [Dialogue]
---
[[pdf]](https://arxiv.org/pdf/2205.09073.pdf)  &emsp;
[[github]](https://github.com/google-research/dialog-inpainting) <br>

**Zhyun Dai<sup>* 1</sup>, Arun Tejasvi Chaganty<sup>* 1</sup>, Vincent Zhao<sup>* 1</sup>, Adia Amini<sup>1</sup>, Qazi Mamunur Rashid<sup>1</sup>, Mike Green<sup>1</sup>, Kelvin Guu<sup>* 1</sup>**
<br><sup>*</sup>Equal Contribution  &emsp; <sup>1</sup>Google Inc. Mountain View, USA. &emsp; 

# Abstract
- 기존 ConvQA 의 scarce training data 문제를 해결하기 위해, document 로 부터 dialogue 를 생성하는 <span style='color:green;font-weight:bold'> dialogue inpainting </span> 방법을 제안한다.
- *dialog inpainting* 방법으로 기존 ConvQA 의 1,000 배 크기의 <span style='color:green;font-weight:bold'> WikiDialog, WebDialog </span> 두 데이터셋을 생성한다.
- 생성한 두 데이터셋을 학습에 활용하여 ConvQA Retreival system 에서 기존 State-of-the-Art model 보다 <span style='background-color: #dcffe4'> 무려 40% 가량의 성능 향상을 보이는 모델 </span>을 제시한다.

# Introduction 
