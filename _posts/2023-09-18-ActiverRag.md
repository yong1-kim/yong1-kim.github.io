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
Generative LM ([GPT-3](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html),[instructGPT](https://arxiv.org/abs/2203.02155),[GPT-4](https://arxiv.org/abs/2212.14024),[PAlm](https://arxiv.org/abs/2204.02311),[RAG](https://arxiv.org/abs/2210.01296),[LLama](https://arxiv.org/abs/2302.13971)) 는 최근 NLP system 에서 foundamental component 이며 언어를 이해하고 생성하는데 있어서 remarkable ability 를 보여준다. 
