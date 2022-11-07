---
layout: post
title:  "[CVPR 2022 Tutorial] Denoising Diffusion-based Generative Modeling: Foundations and Applications(1)"
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
Diffusion model 은 이미 super-resolution, text-to-image generation 에서 매우 강력한 성능을 보여준다.

![image](https://user-images.githubusercontent.com/42200027/200236326-f229a992-035e-45bd-97c6-68c9e0321c26.png)

![image](https://user-images.githubusercontent.com/42200027/200236370-06c622a6-2546-49a9-aa56-a042e5b0942f.png)

# Denoising Diffusion Probabilistic Models

![image](https://user-images.githubusercontent.com/42200027/200237051-48c210cf-4d0a-4cee-861d-5dc653324766.png)

Denoising Diffusion model 은 두 가지 process 로 구성된다.

(1) Forward diffusion process that gradually adds noise to input <br>
(2) Reverse denoising process that learns to generate data by denoising <br>

첫 번째로, forward pass 에 대해서 살펴보면,

![image](https://user-images.githubusercontent.com/42200027/200237508-88d40f19-9dad-4b59-adf6-d08eb441c2b3.png)

위의 그림과 같이, T-step 동안 normal distribution 같은 noise 를 단순하게 추가해주는 방식으로 진행된다.
$\beta$ (noise schedule) 값은 0.0001 정도로 작은 값으로 설정된다. 이후 Join probability 가 Markov Process 로 생성이 된다.

<span style='color:green;font-weight:bold'> Diffusion Kernel </span>

Forward process 는 simple gaussian kernel 의 markov chain 이기 때문에, step 을 건너뛸 수 있다. Diffusion kernel 로 불리는 이 방법은 아래와 같다.
마지막 step 에서는 white noise 만 남게 $\alpha$ 값이 0 이 되게끔 noise schedule 이 design 된다.

![image](https://user-images.githubusercontent.com/42200027/200237949-ddec1851-7081-48a5-ae7c-594b51f6f4a1.png)

지금까지는 conditional disturbition $q(x_t \| x_0 )$ 를 생각했는데, 그렇다면 diffused data distribution $q(x_t)$는 어떻게 정의될까?

![image](https://user-images.githubusercontent.com/42200027/200238673-b178c986-3cd5-4b06-ab84-8126a2be8005.png)

위의 그림에서, input data dist. $x_0$ 에 대해서, 최종 $x_T$ 까지 가는 동안 Diffused data distrubition 이 noise 로 smooth 해지는 것을 볼 수 있다. 
따라서, diffusion kernel은 step 을 진행할 수록 distribution 을 smoother and smoother 하게 해주는 **Gaussian convolution** 이다.

<span style='color:green;font-weight:bold'> Generative Learning by Denoising </span>

이제 반대로, 어떻게 standard normal distribution 에서 sample 을 해서 원하는 data distribution value 를 얻을 수 있을까?
우리는 $q(x_t)$ 의 diffusion dist. 를 가지고 있으므로, 반복적으로 $x_{t-1}$ 를 True Denoising Dist. 를 활용해 sample 하면 된다.
그러나 문제는, 이 denoising distribution 이  **intractable** 하다는 것이다. 즉 다시 말해, 이 dist. 에 access 할 수 없다는 것이다.
이 식에서 $q(x_{t-1})$ 는 미래의 dist. 이기 때문에 접근할 수가 없기 때문이다. 
따라서, 우리가 해야할 것은 **approximation** 이다. 이 때 중요한 것은 each step 의 noise schedule $\beta$ 값이 굉장히 작아야 한다는 것이다. 

![image](https://user-images.githubusercontent.com/42200027/200239381-e24be212-4ea1-4244-ac23-3b72b2b73d17.png)





# Score-based Generative Modeling with Differenital Equations

# Advanced Techniques : Accelerated Sampling, Conditional Generation, and Beyond

# Application (1) : Image Synthesis, Text-to-Image, Controllable Generation

# Application (2) : Image Editing, Image-to-Image, Super-resolution, Segmentation

# Application (3) : Video Synthesis, Medical imaging, 3D Generation, Discrete State Models

# Conclusions, Open Problems
