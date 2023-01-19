---
title: "Meta-Learning이란?(2/2)"
excerpt: "먼저 메타러닝을 공부하기 전 지도학습의 task를 정의해봅시다."
categories:
  - Meta-Learning
tags:
  - [Meta-Learning, ML, Machine Learning, Deep Learning]
use_math: true

toc: true
toc_sticky: true

date: 2022-12-27
last_modified_at: 2022-12-27
---

먼저 메타러닝을 공부하기 전 지도학습의 task를 정의해봅시다.

{:refdef: style="text-align: center;"}
$T={p(x),p(y&#124;x),L}$
{: refdef}

여기서 $p(x)$는 입력의 확률분포, $p(y&#124;x)$는 출력의 확률분포, $L$은 손실함수입니다.

지도학습은 이에 대한 Loss를 최소화하는 파라미터를 찾는 것입니다.

메타러닝에서는 어떻게 해야할까요?

지도학습에서는 $T$를 고정시켜놓고 $p(x)$와 $p(y&#124;x)$를 학습하는 것이지만, 메타러닝에서는 $T$를 학습하는 것입니다.

{:refdef: style="text-align: center;"}
$T={p(x),p(y&#124;x),L} \rightarrow T={p(x),p(y&#124;x),L}$
{: refdef}
{:refdef: style="text-align: center;"}
$T_{1}, T_{2}, \cdots, T_{N} \sim p(T)$
{: refdef}

아직 감이 오지 않습니다.

{:refdef: style="text-align: center;"}
![img](/assets/img/meta_figure3.png)
{: refdef}

[Optimization as a Model for Few-Shot Learning](https://openreview.net/pdf?id=rJY0-Kcll)에서는 위와 같은 그림을 사용합니다.

위 이미지는 크게 $D_{meta-train}$와 $D_{meta-test}$로 나뉩니다.

자세히 그림을 보면 $D_{meta-train}$과 $D_{meta-test}$는 겹치는 label이 존재하지 않습니다.

왜 이렇게 나누었을까요?

우리의 목표는 학습하는 방법을 학습하는 것이었죠?

외우지 않기 위해, Meta-Train에서 학습하는 방법을 학습하고,

Meta-Test에서는 학습한 방법을 사용해야 합니다.

하얀색 박스 하나만 본다면, 그 안에서는 지도학습과 같은 과정이라고 생각할 수 있겠죠?

$D_{train}$만으로 $D_{test}$를 잘 맞추어야 합니다.

이렇게 하얀색 박스가 여럿 존재하는 데이터셋을 **Meta-dataset**이라고 합니다.

{:refdef: style="text-align: center; font-size: 80%; color: gray;"}
참고로 $D_{train}$은 Support Set, $D_{test}$는 Query Set이라고 부르기도 합니다.
{: refdef}

<hr/>
메타러닝의 방법론은 여러가지입니다.
**좋은 초기화(initialization)**, **최적화 프로세스(optimization process)**, **좋은 임베딩(embedding space)**를 찾는 방법 등이 있습니다.

$D_{meta-train} = {(D_{1}^{\, train}, D_{1}^{\,test}), (D_{2}^{\,train}, D_{2}^{\,test}), \cdots, (D_{N}^{\,train}, D_{N}^{\,test})}$

$D_{meta-train}$은 여러개의 task로 이루어져 있습니다.

여기서 $D^{train}$과 $D^{test}$은 다음과 같이 나타낼 수 있죠,

$D_{i}^{\,train} = {(x_{1}, y_{1}), (x_{2}, y_{2}), \cdots, (x_{n}, y_{n})}$
<br/><br/>
$D_{i}^{\,test} = {(x_{n+1}, y_{n+1}), (x_{n+2}, y_{n+2}), \cdots, (x_{m}, y_{m})}$

수식과 공부한 것을 같이 이해하면 메타 트레인 $D_{meta-train}$의 하나의 task $T_{i}$는 $D_{i}^{\,train}$과 $D_{i}^{\,test}$로 이루어져 있습니다.

그럼 최적의 parameter는 다음처럼 학습할 수 있습니다.

$\theta^{\,*} = \arg \max_{\theta} \log p(\theta &#124; D_{meta-train})$

최적의 paramter인 $\theta^{\,*}$를 학습했다고 합시다.

이제 $\theta^{\,*}$를 사용해 $D_{meta-test}$에서 테스트를 진행해야합니다.

$\phi^{*} = \arg \max_{\phi} \log p(\phi &#124; D_{meta-test}, {\theta})$

수식이 어려운 당신들을 위해... 해석해봅시다.

$\theta^{\,*}$는 $D_{meta-train}$에서 학습한 최적의 parameter입니다.

수식에서 $(left &#124; right)$는 $left$가 주어졌을 때 $right$가 나올 확률을 의미합니다.

그러면, 학습한 $\theta^{\,*}$를 사용해 $D_{meta-test}$에서 테스트를 진행하고,

이때 $\phi$중 가장 높은 스코어를 내는(argmax) $\phi^{*}$를 찾는다는 의미입니다.

다음장부터 메타러닝이 어떻게 학습하는지 알아봅시다.
