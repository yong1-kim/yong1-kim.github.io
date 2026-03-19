---
layout: project_single
title:  "Findings of NAACL 2025"
slug: "NAACL2025-Findings"
---
**SWITCH: Studying with Teacher for Knowledge Distillation of Large Language Models**

Jahyun Koo, Yerin Hwang, **Yongil Kim**, Taegwan Kang, Hyunkyung Bae, Kyomin Jung

**Abstract:**
Despite the success of Large Language Models (LLMs), they still face challenges related to high inference costs and memory requirements. To address these issues, Knowledge Distillation (KD) has emerged as a popular method for model compression, with student-generated outputs (SGOs) as training data being particularly notable for reducing the mismatch between training and inference. However, SGOs often produce noisy and biased sequences, which can lead to misguidance from the teacher model, especially in long sequences. To mitigate these challenges, we propose SWITCH (Studying WIth TeaCHer for Knowledge Distillation), a novel approach that strategically incorporates the teacher model during the student's sequence generation. SWITCH identifies discrepancies between the token probabilities of the teacher and student models, allowing the teacher to intervene selectively. Extensive experimental results across three model families and five instruction-following datasets show that SWITCH surpasses traditional KD methods, particularly excelling in the generation of long sequential data.

[\[Paper\]](https://arxiv.org/abs/2410.19503)
