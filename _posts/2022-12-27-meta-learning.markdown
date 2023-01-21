---
title: "Meta-Learning이란?(1/2)"
excerpt: "메타러닝이 정확히 어떤 것인지 알아보자"

categories:
  - Meta Learning
tags:
  - [Meta-Learning, ML, Machine Learning, Deep Learning]
use_th: true

toc: true
toc_sticky: true

date: 2022-12-27
last_modified_at: 2022-12-27
---

딥러닝의 발전의 근본적인 이유 중 하나는 크고 다양한 데이터 셋과 좋은 하드웨어의 발전이 있었습니다.

여기서 딥러닝의 단점이 드러납니다. 크고 다양한 데이터 셋과, 비싼 GPU등이 필수적입니다.

사람의 뇌와 달리, 적은 데이터로 빠르게 학습하기 어렵다는 것입니다.

그렇다면, 사람처럼 적은 데이터로 빠르게 학습할 수 있는 방법은 없을까요?

이 문제를 해결하고자 하는 분야가 **Meta-Learning**입니다.

인공지능에서의 Meta-Learning은 **학습을 위해 학습(learning to learn)**을 의미합니다.

즉, 새로운 task를 더 빨리 학습하기 위해 이전의 학습 경험을 적극적으로 활용합니다.

데이터를 학습하는 것뿐만 아니라 **자신의 학습 능력** 또한 스스로 향상시킵니다.

CNN의 inductive bias는 이미지의 특징을 추출하는데 특화되어 있습니다.
Meta-Learning의 inductive bias는 **새로운 task를 빠르게 학습하는 능력**을 추출하는데 특화되어 있습니다.

{:refdef: style="text-align: center;"}
![img](/assets/img/meta_figure1.png)
{: refdef}

왼쪽 multi-task learning은 task마다 다른 모델을 학습합니다.
반면, 오른쪽 **meta-learning**은 task마다 다른 모델을 학습하는 것이 아니라, **task를 학습하는 모델**을 학습합니다.
테스트 시 학습 때 보지 못했던 새로운 태스크가 주어졌을 때, meta-learning은 더 빠르게 학습하는 것을 목표로 합니다.

{:refdef: style="text-align: center;"}
![img](/assets/img/meta_figure2.png)
{: refdef}
왼쪽에는 Training data가 주어지고, 오른쪽에는 test data가 주어졌습니다.
사람이 보기에는 당연히 Braque가 그린 것이라고 판단할 수 있죠?(조금 어렵나요)

우리는 세상을 살아오며, 많은 시각정 정보들로 **지식**을 축적합니다.

이를 통해 많은 학습을 하며, 사전 지식을 충분히 가지고 있습니다.

따라서 우리는 이 사전 **지식**을 통해 이미지를 분류할 수 있었습니다.

이러한 문제를 **few-shot learning**이라고 합니다.

문제를 더 구체적으로 정의하면 N-way K-shot learning이라고 합니다.

위 사진에서는 class가 2개고, 클래스별로 3개의 이미지를 주었으므로, 2-way 3-shot learning입니다.

인공지능은 이를 쉽게 할 수 없죠.

이를 해결하기 위해 **Meta-Learning**이 등장합니다.

[모두를 위한 메타러닝](https://wikibook.co.kr/meta)책을 참고하였습니다.
