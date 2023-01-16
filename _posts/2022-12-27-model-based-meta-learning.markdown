---
layout: post
title: "모델 기반 메타러닝(NTM, MANN, SNAIL)"
date: 2022-12-27 18:23:59 +0900
categories: RL ML Meta-Learning
tags: [Meta-Learning, ML, Machine Learning, Deep Learning]
use_math: true
---

본 포스트는 LSTM에 대한 사전 지식을 필요로 합니다.

{:refdef: style="text-align: center;"}
![img](/assets/img/lstm_figure.png)
{: refdef}

모델 기반 메타 러닝에서 학습하고자 하는 학습 방법은 RNN의 은닉상태와 같은 내부 parameter이며, 이를 잘 학습하는 것이 목쵸입니다.

모델 기반 메타 러닝에서 핵심 아이디어는 우리가 원래 아는 LSTM같은 신경망을 학습시키는 것입니다.

LSTM같은 순환 신경망은 일종의 메모리 역할을 하며,

이를 통해 아! LSTM을 통해 많은 task들을 학습하는 것이 이 정보를 기억하는구나! 라고 해석할 수 있습니다.

## 1. Neural Turing Machine(NTM)

