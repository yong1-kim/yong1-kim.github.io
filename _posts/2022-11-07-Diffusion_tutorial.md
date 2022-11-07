---
layout: post
title:  "[CVPR 2022 Tutorial] Denoising Diffusion-based Generative Modeling: Foundations and Applications"
date:   2022-11-05 19:12:00 +0900
use_math: true
categories: [Diffusion]
---
[[blog]](https://cvpr2022-tutorial-diffusion-models.github.io/)
[[youtube]](https://www.youtube.com/watch?v=cS6JQpEY9cs)

**Karsten Kresis<sup>1</sup>, Ruiqi Gao<sup>2</sup>, Arash Vahdat<sup>1</sup>**
<br><sup>1</sup> NVIDIA <sup>2</sup> Google Brain &emsp; 

<span style='background-color: #dcffe4'> 이 포스트는 CVPR2022 Tutorial : Denoising Diffusion-based Generative Modeling 을 기반으로 작성한 내용을 담고 있습니다. </span>
  
# Deep Generative Learning

<span style='color:green;font-weight:bold'>  Learning to generate data  </span>
Generative model 은 data distribution 으로 부터 학습(train)한 후, 추론(inference) 시에 하나의 sample 을 generation 하는 모델을 의미한다.
Generative model 은 Content Generation, Representation Learning, Artistic Tools 등에서 이미 굉장히 좋은 성능을 보이고 있다. 
![image](https://user-images.githubusercontent.com/42200027/200235662-2ae7c31d-5928-491d-a34f-3ad2343b8492.png)

현재까지 GAN(Generative Adversarial Networks) 를 필두로, VAE(Variational Autoencoders), Energy-based models, Autoregressive models, 그리고 Normalizing Flows 에 이르기까지 Computer Vision 분야에서 많은 generative model 이 연구되어 왔지만, <span style='color:green;font-weight:bold'> 새롭고 강력한 (new and strong) Denoising Diffusion Models</span> 가 이들을 섭렵해갈 것이라고 예상하고 있다.

![image](https://user-images.githubusercontent.com/42200027/200235994-df851d3d-02e4-4f89-8055-d4ea4602d739.png)

그림에서 볼 수 있듯이, 최근 연구되는 Denoising Diffusion model 은 ImageNet 과 같이 Challenging 한 dataset 들에 대해서도 굉장히 좋은 퀄리티의 이미지를 생성할 수 있고, 또 다양한 결과를 내어놓는다. 왼쪽은 openAi 에서, 오른쪽은 Google 에서 연구된 최신 diffusion model results 이다. 이것들은 GAN 을 뛰어넘는 성과를 보였다.
