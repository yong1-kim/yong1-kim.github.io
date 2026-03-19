---
layout: project_single
title:  "Findings of EMNLP 2025"
slug: "EMNLP2025-Findings-Trick"
---
**Can You Trick the Grader? Adversarial Persuasion of LLM Judges**

Yerin Hwang, Dongryeol Lee, Taegwan Kang, **Yongil Kim**, Kyomin Jung

**Abstract:**
As large language models increasingly serve as automated evaluators, we investigate whether persuasive language can bias LLM judges when scoring mathematical tasks. Drawing from Aristotle's rhetorical principles, we test seven techniques: Majority, Consistency, Flattery, Reciprocity, Pity, Authority, and Identity. Embedding these into otherwise identical responses, we find that persuasive language causes inflated scores for incorrect solutions across six math benchmarks, averaging up to 8% higher, with Consistency producing the most severe distortion. Notably, larger model sizes do not substantially mitigate this vulnerability. Combined techniques amplify bias further, and pairwise evaluation proves equally susceptible. The persuasive effect persists even with counter-prompting strategies, revealing a critical vulnerability in LLM-as-a-Judge systems and highlighting the need for robust defenses against persuasion-based attacks.

[\[Paper\]](https://arxiv.org/abs/2508.07805)
