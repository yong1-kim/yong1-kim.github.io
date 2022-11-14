---
layout: post
title:  "[ICML2022] Data Determinces Distributional Robustness in Contrastive Language-Image Pre-training (CLIP)"
date:   2022-11-14 16:30:00 +0900
use_math: true
categories: [Transformer, Vision-and-Language]
---
[[pdf]](https://proceedings.mlr.press/v162/fang22a/fang22a.pdf)  &emsp;

**Alex Fang<sup>1</sup>, Gabriel Ilharco<sup>1</sup>, Mitchell Wortsman<sup>1</sup>, Yuhao Wan<sup>1</sup>, Vaishaal Shankar<sup>2</sup>, Achal Dave<sup>2</sup>, Ludwig Schmidt<sup>1 3</sup>**
<br><sup>1</sup>University of Washington ,<sup>2</sup> Amazon, <sup>3</sup> Allen Institute for Artificial Intelligence. &emsp; 

![image](https://user-images.githubusercontent.com/42200027/201605458-96e586c0-2ca1-4ce9-8e81-7e895c3d732d.png)

# Abstract
- (Motivation) CLIP, ALIGN, BASIC 과 같은 contrastive learning 기반의 vision-language model 들은 distribution shift 에 굉장한 robustness 를 보인다. 이렇게 큰 robustness gain 을 얻는 원인에 대한 질문은 굉장히 중요하다.
- (Solution) 체계적인 실험 조사(systematic experimental investigation) 으로 이 질문에 대해 탐구한다. 
- (Method) (1) Training set size (2) Training distribution (3) Language supervision at training time (4) Language supervision at test time (5) contrastive loss function 다섯 가지 possible cause 에 대해서 실험 조사를 진행한다.
- (Result) (2) Training distribution 이 다양할 수록 robustness gain 이 컸고, 나머지 네 개의 factor 들은 전혀 robustness 에 관련이 없었다.
- (New Dataset) Flickr annotation 으로  이뤄진 ImageNet version 의 새로운 dataset 인  **ImageNet-Captions** 을 공개한다. 이 데이터셋은 controllable vision-and-language training 이 가능하게 한다.

# Introduction
[CLIP](http://proceedings.mlr.press/v139/radford21a.html), [ALIGN](https://arxiv.org/abs/2102.05918), [BASIC](https://arxiv.org/abs/2111.10050) 과 같은 vision-and-language large pretrained model 은 다양한 natural distribution shift 에 전례없는 굉장한 robustness 를 보인다. 
기존의 모델들이 class annotation 과 함께 image 를 학습한 것에 대조적으로, CLIP 과 그 relative 들은 image 와 그에 상응하는 web 에서 얻은 unstructured text 를 직접적으로 학습한다. 
이러한 모델들은 [ImageNetV2](https://arxiv.org/abs/1902.10811), [ObjectNet](https://proceedings.neurips.cc/paper/2019/file/97af07a14cacba681feacf3012730892-Paper.pdf) 과 같은 어려운 distribution shift 에서 large robustness 를 달성한다
그동안은, Machine Learning 기법의 숱한 발전에도 이 데이터셋들에 대해 이 정도의 향상된 robustness 를 보였던 알고리즘 기술이 없었다.
따라서 중요한 질문이 떠오른다 : <span style='background-color: #dcffe4'> "What causes CLIP's unprecendted robustness?" </span>

Vision 하나만의 기술이 아니라, Language-image model (vision-and-language model) 이 처음으로 large robustness gain 을 성취해냈다는 사실에서, language and image multimodal learning 이 robustness 의 key 가 될 것이라고 예상할 수 있다.
그러나 CLIP의 robustness 의 원인을 pinpoint 하기는 굉장히 어려운데, 그 이유는 CLIP 이 기존의 image classification model 의 학습 패러다임에서 꽤 많은 여러 변화를 통해 학습되었기 때문이다.
예를 들어, 높은 accuracy 를 보이는 CLIP model 은 Vision Transformer ([ViT](https://arxiv.org/abs/2010.11929)) 구조를 통해 학습이 된다. 
그러나 [Radford et al.](http://proceedings.mlr.press/v139/radford21a.html) 은 CLIP 논문에서 이미 model architecture 와 size 에 대해서 조사를 했고, 이러한 요소들은 robustness 에 크게 관여하지 않는다는 것을 밝혀냈다.
그럼에도 불구하고, 다음의 여러가지 가능성 높은 요소들이 CLIP 의 robustness 의 원인이 될 수 있다.
- The large training set size (400 million images)
- The training distribution
- Language supervision at training time
- Language supervision at test time via prompts
- The contrastive loss function

CLIP 의 robustness 를 이해하는 것은 앞으로 reliable machine learning 을 guide 해줄 수 있는 방향을 제시해 주기 때문에 매우 중요하다.

이 논문에서는 위의 제시된 다섯가지 가능성 높은 원인들에 대해 controlled experiment 를 통해 CLIP 의 robustness 의 원인을 밝혀낸다.
<span style='background-color: #dcffe4'> Main result 는 CLIP 의 robustness 는 training distribution 에 의해 결정된다는 것이다. </span>
Training time 에서의 Language supervision 은 standard supervised learning 에 비해 model 을 더 robust 해지게 만들지 않는다.
따라서 Language supervision 은 robustness 에 *indirect* effect 만 미치고 있다.
상세하게는, language supervision 은 class label 의 consistent annotation 의 필요성을 제거하게 도와주어, image의  diverse distribution 을 간단하게 학습할 수 있도록 도와준다.
다시 한 번 결론은, <span style='color:green;font-weight:bold'> The more diverse training distribution –– not the language supervision –– then leads to more robust representations. </span> 이다.

CLIP robustness 에 대한 조사를 위한 연구 방향으로 크게 두 가지 방향으로 정리할 수 있다.
<span style='background-color: #dcffe4'> 첫 번째는, 새로운 데이터셋 ImageNet-Captions 의 소개이다. </span>
ImageNet-Captions 는 paired language-image data 로, 120만개의 [ImageNet 2012 training set](https://arxiv.org/abs/1409.0575) 중 463,622 개의 image 를 original text data 를 augmentation 하여 생성하였다. original text data 는 상응하는 Flickr image 로 부터 추출한다.
ImageNet-Captions 은 같은 image 를 통해, 기존의 standard ImageNet training 과, language-image training 두 가지 학습 방법을 controlled experiment 로 비교할 수 있게 도와준다.

<span style='background-color: #dcffe4'> 두 번째로, CLIP training 과 성능은 유사하지만, vision component 와 language component 사이의 interaction 은 최소화하는,새로운 language-image training 을 위한 baseline 을 소개한다.</span>
특히, 아래의 training procedure 를 소개하고, [YFCC-15M dataset](https://www.arxiv-vanity.com/papers/1503.01817/) 에 대해 그 행동을 illustrate 한다. 

(1) YFCC-15M 의 image 만(*only image*) 을 pre-train 하기 위해 [SimCLR](http://proceedings.mlr.press/v119/chen20j.html) 을 사용.

(2) Simple *text match* 를 통해, ImageNet class 와 YFCC-15M sample 을 matching 하여 (1) 의 resulting representation 을 fine-tuning.

특히, 저자들의 이러한 접근은 **language model 에 의존하지 않기** 때문에, 훨씬 단순한 언어 처리로 CLIP training 과 유사한 성능을 가져갈 수 있다.
CLIP training 을 이해하기 위한 baseline 제공을 넘어서, 저자들의 이러한 단순한 어프로치가 language-image trainig 에 대해 알고리즘적인 개선에 대한 길을 터주었다고 말하고 있다.

# Background
CLIP 의 robustness 의 원인을 pinpoint 하기 위해서는 다양한 모델에 대한 robusntess 비교를 위한 precise 한 experimental setup 이 필요하다.
우선, [Taori et al.](https://arxiv.org/abs/2007.00644) 에 의해 소개된 *effective robustness framework* 를 background 로 살펴보고, CLIP model 의 robustness gain 에 대해서 실험해본다. 


<span style='color:green;font-weight:bold'> Experimental setup for measuring robustness </span>
<br> 
Reliable machine learning model 을 만든다는 것은 diverse range of test distribution 에 대해 잘 작동하는 모델을 디자인하는 것을 의미한다.
예를 들어, imageNet 에서 75% 의 accuracy 를 보인다면, 그와 유사한 데이터셋인 ImageNetV2 에 대해서도 (인간이 그러하듯) 75% 와 유사한 성능을 보여야 한다. ([[1]](https://proceedings.mlr.press/v119/shankar20c.html))
그러나, 이러한 consistent performance 를 보이지 않고, 대부분의 모델은은 이 distribution shift 에 대해 12 percepnt point 의 성능 drop 을 보인다. ([[2]](https://arxiv.org/abs/1902.10811)) 
반면, [Radford et al.](http://proceedings.mlr.press/v139/radford21a.html) 에서 제시되는 CLIP model 은 단지 6 percent point 만의 drop 을 보여 robustness 를 갖는다.
ImageNet 에서 뿐 아니라, 다른 많은 distribution shift 에 대해서도 CLIP 은 훨씬 더 적은 accuracy drop 을 보인다.
(**여기서의 CLIP 은 Radford 의 CLIP model 이 아니라 contrastive learning 기법으로 vision-language task 를 학습한 ALIGN, BASIC 등의 모델을 포함한 기법을 말한다**)

수식적으로, model $f$ 와 두 test distribution $D_1$, $D_2$ 에 대하여, $acc_{D_1}(f)$ 와 $acc_{D_2}(f)$ 를 측정하여 비교한다.
보통 $D_1$ 은 ImageNet (ILSVRC-2012) test set 이 되고, $D_2$는 여러가지 다른 out-of-distribution test set 이 된다.
당연히 ideal model 은 두 distribution 에서 100% accuracy 를 보이는 것이지만, 그러한 모델은 존재하지 않기 때문에, 두 accuracy 의 차이가 없는, robustness 를 가지는 것에 대해서 모델 비교를 진행한다.
한 가지 confounder 는 $D_1$ 에 대한 accuracy 가 증가하면 $D_2$에 대한 accuracy gain 이 이미 증가해있다는 것이다. ([[3]](https://arxiv.org/abs/2007.00644),[[4]](https://arxiv.org/abs/2107.04649))
위의 Figure 1 에서, 파란색 점은 imageNet 으로 학습된 모델들이다. x 축은 $acc_{D_1}(f)$ 이고, y 축은 $acc_{D_2}(f)$ 이다. 4 개의 out-of-distribution shift 에 대한 평균값이 y 축 값에 해당한다. 
파란색 점으로 scatter 된 ImageNet 으로 학습된 모델들을 보면, (또 다른 모든 모델들에 대해서도) ImageNet accuracy 를 높이는 덕목만으로도 다른 distribution shift 에 대한 accuracy 역시 높아졌다. (우상향 했다)

Robustness 측정단계에서, 이러한 교란 인자(confounder)를 처리하기 위해, [Taori et al.](https://arxiv.org/abs/2007.00644) 은 robustness 에 대한 정의를 *accuracy beyond the baseline* given by ImageNet models 로 했다. 
그 논문의 저자들은 이 것을 quantity *effective robustness* 라고 칭한다.
Figure 1 에서 파란색 선에서 수직으로 뻗는 *Effiective Robustness* 가 그것이다.
[Radford et al.](http://proceedings.mlr.press/v139/radford21a.html) 은 Figure 1 의 purple line 처럼 high effective robustness 를 달성한 CLIP model 을 구현했다고 증명한다.
수식적으로, 이 effective robustness 비교는 다음의 식으로 표현가능하다.
Baseline fucntion $\beta$ : $R -> R$ 에 대해, $\beta$는 $acc_{D_1}(f)$ 으로부터 $acc_{D_2}(f)$ 로 mapping 하는 함수이다.
New model $f'$ 에 대하여, effective robustness 는 다음과 같이 표시할 수 있다.
$\rho(f') = acc_{D_2}(f') - \beta(acc_{D_1}(f'))$.
이 수식이 이 논문에서 CLIP model 들의 robustness 를 이해하기 위해 visualize 하는 main quantity 이다.

기존의 [Taori et al.](https://arxiv.org/abs/2007.00644) 과 [Radford et al.](http://proceedings.mlr.press/v139/radford21a.html) 에서와 마찬가지로, *natural distribution shift* 에 집중하여 실험을 진행한다.
Natural variation 은 lighting, geographic location 등을 포함하는 것으로, *synthetic* distribution shift 와 반대되는 개념이다.
*Synthetic* distribution shift 는 인위적으로 computationally modification 을 준 것으로, Gaussian noise 부여, blur 부여, perturbation 부여 등이 속한다.
Natural distribution 은 real data 를 표방하기 때문에, 아래의 natural distribution shift dataset 을 선정한다.

(1) ImageNet-V2 ([Recht et al., 2019](https://arxiv.org/abs/1902.10811)) : a reproduction of the ImageNet validation set with distribution shift due to changes in the crowdsourcing process.

(2) ImageNet-Sketch ([Wang et al., 2019](https://arxiv.org/abs/1905.13549)) : black and white sketches of ImageNet images.

(3) ImageNet-R ([Hendrycks et al., 2021](https://arxiv.org/abs/2006.16241)) : renditions (e.g., art, patterns, etc.) of 200 ImageNet classes.

(4) ObjectNet ([Barbu et al., 2019](https://proceedings.neurips.cc/paper/2019/file/97af07a14cacba681feacf3012730892-Paper.pdf)) : real-world objects from ImageNet with crowd-sourced random backgrounds, rotations, and viewpoints

(5) ImageNet-A ([Hendrycks et al., 2019](https://arxiv.org/abs/1907.07174)) : naturally occurring examples filtered so they are misclassified by a ResNet-50 model.

이러한 distribution shift 로의 effective robostness 의 중요한 property 는 **training set 의 size 가 달라진다고해서 effective robustness 에는 영향이 없다** 는 것이다.
[Taori et al.](https://arxiv.org/abs/2007.00644) 과 [Miller et al.](https://arxiv.org/abs/2107.04649) 에서는 이미 training set 의 sub-sampling 이 accuracy 에는 영향을 주지만, effective robustness 에는 전혀 영향이 없다는 것을 증명하였다. 
<span style='background-color: #dcffe4'> 이 것으로 CLIP 의 high effective robustness 에 대해 training set size 는 rule out 된다. </span>

<span style='color:green;font-weight:bold'> Additional related work </span>
<br> 
기존의 [VirTex](https://openaccess.thecvf.com/content/CVPR2021/html/Desai_VirTex_Learning_Visual_Representations_From_Textual_Annotations_CVPR_2021_paper.html), [ICMLM](https://link.springer.com/chapter/10.1007/978-3-030-58598-3_10), [ConVIRT](https://arxiv.org/abs/2010.00747) 와 같은 Vision-language model 이 활발히 연구되어 왔지만, CLIP 과 [ALIGN](https://arxiv.org/abs/2102.05918) 은 굉장히 큰 corpus 에 대해서 학습을 하고, 많은 downstream task 에서 좋은 성능을 보였으며, 전례없는 강한 robustness 를 보유한 모델이다.

CLIP 의 generalization 성능에 대해서 분석을 하는 연구들도 있었다.
[Devillers et al.](https://arxiv.org/abs/2104.08313) 은 CLIP 과 같은 multimodal model 이 few-shot 과 linear probe 결과를 통해 좋은 generalization 성능을 보이는 것에 대해, image 와 text 두 modality 중 하나만을 사용하여 실험을 진행하였다. 실험 분석 결과, 하나의 modality 만을 사용했을 때에 비해  multimodal model 의 이점이 딱히 드러나지 않았다. 
반면 우리는 CLIP 의 robustness 에 대하여 language 가 어떻게 out-of distribution generalization 에영향을 주는지를 연구한다. 
기존 Devillers et al. 과의 차이점은, 본 연구에서는 accuracy 와 robustness 를 구분하기 위해, in-distribution accruacy 를 control 해서 비교한다는 것이다.

[Anderassen et al.](https://arxiv.org/abs/2106.15831) 에서는 fine-tuning process 가 진행될 수록, CLIP 의 zero-shot capability, effective robustness 가 줄어든 것을 확인한다.
Radford et al. 의 CLIP 이후 ALIGN, BASIC, [LiT](https://arxiv.org/abs/2111.07991) 등의 유사한 논문이 많이 나왔지만, 본 연구와 가장 유사한 연구는  LiT 이다.
LiT는 pre-trained image model 을 사용하고, downstream task 에 대해 text head 만을 fine-tuning 하여 좋은 성능을 얻는 모델이다.
본 연구가 LiT 와 가장 다른 점은 LiT는 zero-shot 성능을 얻기 위해 4 billion image-caption pair 를 fine-tuning 하지만, 본 연구에서는 substring matching 을 통해 caption 을 class label 로 바꾼  후, regular image classifier 를 통해 학습한다는 차이점이 있다.

# ImageNet-Captions
![image](https://user-images.githubusercontent.com/42200027/201620183-cf09d1ea-c79b-4130-b969-fddf620a3062.png)

저자들은 image-text supervision 을 위한 실험을 위해 새로운 데이터셋인 ImageNet-Captions 를 만들었다. 
다음의 네 가지 요구에 의해 ImageNet-Caption 을 생성하였다.

- Effective robustness 에 자연어 supervision 의 효과를 isolate 하기 위해, 자연어 supervision 에 더불어 traditional classification label 도 함께 있는 데이터셋이 필요했다. 이 두 label 은 구조적인 차이를 전혀 발생시키지 않고, solely 다른 loss function 만을 통해 다른 모델이 학습되게 실험을 설계할 수 있게 도와준다. 
- Synthetically 생성된 caption 대신 original image source 로부터 오는 text annotation 이 필요하다. (model bias 를 없애준다)
- ImageNet 과 같은 흔히 사용되는 benchmark 와 연관되어 있어야 한다.
- 최신 연구에 걸맞는 충분히 큰 사이즈여야 한다.

이 연구전에 이러한 점들을 모두 만족하는 데이터셋이 없었다.
ImageNet-Captions 은 ImageNet (ILSVRC 2012) training set 의 subset이고, Flickr 로부터 얻은 paired original image title/description/tag 을 갖고 있다. (ImageNet 은 대부부 Flickr 로부터 생성되었다). 

<span style='color:green;font-weight:bold'> Constructing ImageNet-Captions </span>
<br> 
ImageNet-Captions 의 목표는, ImageNet iamge 에 original text data 를 augment 하는 것이다.
그러나 ImageNet 2012 dataset 에는 어떠한 metadata 도 없어서 그것이 쉬운 일은 아니다.
저자들은 다음 세 가지 fact 로 부터 데이터셋을 구성한다:

- ImageNet 의 대부분은 Flickr 로부터 생성되었다.
- Imagnet fall 2011 은 URL 을 가지고 있다.
- Photo identifier 를 통해, Flickr API 가 associated text data 를 제공할 수 있다.

저자들은 image URL 을 통해 Flickr 에 속해있는 ImageNet fall 11 dataset 을 추려낸 후, 1 천개의 class label 로 제한하여 64만 개의 데이터를 추려냈다.
이후, [Jain et al.](https://github.com/idealo/imagededup) 의 중복 제거 (deduplication) 방법을 통해 ILSVRC 2012 에 없는 image 를 제거했다.
또, profanity(불경스러운 단어)를 포함한 image 를 제거하니, 463,622 개의 image 가 추려졌다.
이 것은 이제 ILSVRC-2012 의 subset 이면서, original text data 를 갖고 있다.
특히, 이 text data 는 title/description/class label 을 포함하고 있다.

<span style='color:green;font-weight:bold'> Properties of ImageNet-Captions </span>
<br> 
![image](https://user-images.githubusercontent.com/42200027/201622331-07a5baa1-2a55-4198-bd63-29400b668763.png)

ImageNet-Captions 은 90% 이상은 영어지만, 127개의 다른 언어도 포함하고 있다.
그리고 위의 표에서와 같이, 94% 의 경우에서 class label 이 corresponding text 에 포함되어 있다.
따라서, ImageNet-Captions 의 caption 들이 class 에 relevant information 을 포함하고 있고, image-text model 의 training 에 적합한 좋은 caption 을 갖고 있다는 것을 알 수 있다.

# Imagenet-Captions experiments 
