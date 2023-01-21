---
title: "Decision Transformer code review(1/2)"
excerpt: "Decision Transformer의 구조와 학습 과정을 코드 리뷰로 살펴봅니다."
categories:
  - RL
tags:
  - [Decision, Transformer, pytorch, RL, ML, code review]
use_math: true

toc: true
toc_sticky: true

date: 2022-12-21
last_modified_at: 2022-12-21
---

이 code review는 다음 [colab code](https://colab.research.google.com/github/nikhilbarhate99/min-decision-transformer/blob/master/min_decision_transformer.ipynb#scrollTo=3uycTGiqjKYK)를 기반으로 합니다.

[D4RL](https://arxiv.org/pdf/2004.07219.pdf)은 Datasets for Deep-Dricen Reinforcement Learning의 줄임말로,

Deep Reinforcement Learning을 위한 데이터셋을 제공합니다.

Maze2D, AntMaze, Adroit, **Gym**, Flow, CARLA 등 다양한 환경에서 수집한 데이터셋을 제공합니다.

```python
dataset = "medium"       # medium / medium-replay / medium-expert
rtg_scale = 1000                # scale to normalize returns to go
```

이 [min decision transformer code](https://colab.research.google.com/github/nikhilbarhate99/min-decision-transformer/blob/master/min_decision_transformer.ipynb#scrollTo=3uycTGiqjKYK)에서는 [D4RL](https://arxiv.org/pdf/2004.07219.pdf)의 Walker2d 환경에서 데이터를 수집하고, Decision Transformer를 학습시킵니다.

Decision Transformer에서 dataset을 다음과 같이 소개합니다!

**medium** datasets은 1백만 steps 후에 생성된 정말 "medium" dataset이며 전문가 policy의 약 1/3정도의 스코어를 냅니다.

**medium-replay** datasets은 medium policy를 통해 학습된 policy가 생성한 dataset(이 환경에선 25k-400k steps의 데이터셋이라 합니다.)

**medium-expert** datasets은 전문가의 시연과 suboptimal data, 부분 학습된 policy나 random policy가 섞인 데이터셋입니다.

```python
env_name = 'Walker2d-v3'
rtg_target = 5000
env_d4rl_name = f'walker2d-{dataset}-v2'
```

{:refdef: style="text-align: center;"}
![image](/assets/img/walker2d.gif)
{: refdef}

{: style="color:gray; font-size: 80%; text-align: center;"}
gym 환경에서의 Walker2d
{: refdef}
<br/><br/>
가장 기초가 되는 Transfomer의 구조부터 살펴봅시다.

```python
class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)
```

\**n_heads*는 multi-head attention에서 head의 개수를 의미합니다.

head의 개수가 많을수록 병렬적으로 attention을 많이 계산할 수 있습니다.

병렬적으로 attention으로 계산하는 것은 다양한 시각으로 정보를 학습할 수 있다고 해석 가능합니다!

q_net, k_net, v_net은 각각 query, key, value를 위한 fully connected layer(h_dim -> h_dim)입니다.
여기서 h_dim은 hidden dimension을 의미합니다.

<br/><br/>
forward 코드를 계속해서 봅시다.

```python
def forward(self, x):

        B, T, C = x.shape # batch size, seq length, attention_dim * n_heads

        # N = num heads(병렬 attention = 얼마나 다양한 시각으로 볼 것인지), D = attention_dim
        N, D = self.n_heads, C // self.n_heads

        # rearrange q, k, v as (B, N, T, D) ->(Batch_size, num_heads, seq_length, attention_dim)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out
```

<br/><br/>
일단 batch_size는 빼고 생각해봅시다!

{:refdef: style="text-align: center;"}
![image](/assets/img/DT_code1.png)
{: refdef}

input으로 넣어주는 raw data는 그림과 같습니다.

가로축은 time step 수만큼의 sequence length를 의미하고, 새로축은 hidden dimension을 의미합니다.

현재 새로축이 h_dim \* num_heads로 나타나져 있는데,

q_net, k_net, v_net을 통과하면서 Sequnce length(가로)와 hidden dimension(세로)이 유지됩니다.

이는 각 network가 network를 통과하면서 query, key, value를 만들어내도록 학습되겠다고 해석할 수 있습니다.

```python
# rearrange q, k, v as (B, N, T, D) ->(Batch_size, num_heads, seq_length, attention_dim)
q = self.q_net(x).view(B, T, N, D).transpose(1,2)
k = self.k_net(x).view(B, T, N, D).transpose(1,2)
v = self.v_net(x).view(B, T, N, D).transpose(1,2)
```

<br/><br/>
이제 새로축인 hidden dimension을 num_heads와 attention_dim으로 나누어줍시다.

num_heads는 병렬적으로 attention을 계산할 때 몇개의 head를 사용할 것인지를 의미하고,

attention_dim은 각 head에서 attention을 계산할 때 사용할 dimension을 의미합니다.

쉽게 생각하면, 각 병렬 attention에서의 hidden dimension이라고 생각하면 됩니다.

{:refdef: style="text-align: center;"}
![image](/assets/img/DT_code2.png)
{: refdef}

다른 색깔의 block이 각각 다른 attention을 계산하게 됩니다.

총 3개가 나오게 되죠?
하나의 input sequence를 각각 query, key, value에 대해 network를 통과시켰기 때문입니다.
그래서 각 층마다 query, key, value가 나오게 됩니다.

```python
# weights (B, N, T, T)
weights = q @ k.transpose(2,3) / math.sqrt(D)
```

이제 각 병렬 attention에서의 query와 key를 곱해줍니다.

query는 attention을 계산할 때 사용할 정보를 담고 있고, key는 attention을 계산할 때 참고할 정보를 담고 있습니다.

쉽게 말해서 query가 이거랑 관련있는 놈 누구야! 하면 key가 관련있는 놈들을 찾아줍니다.(관련이 있으면 코사인 유사도(행렬곱)이 크니까!)

```python
# causal mask applied to weights
weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
# normalize weights, all -inf -> 0 after softmax
normalized_weights = F.softmax(weights, dim=-1)
```

아- 벌써 코드가 어려워요

예시를 들어가며 한번 이해해봅시다.

```python
import torch
weights = torch.tensor([[1,1,1],[1,1,1],[1,1,1]], dtype=torch.float32)
ones = torch.ones((3,3))
mask = torch.tril(ones).view(1, 3, 3)
# mask = [[1,0,0],
#         [1,1,0],
#         [1,1,1]]
```

저는 논문과 비슷하게 다음과 같이 예시를 들어보았습니다.

```
weights = weights.masked_fill(mask[...,:3,:3] == 0, float('-inf'))
```

그리고 masking을 해주기 위해 masked_fill을 사용해주었습니다.

mask가 0인 부분은 -inf로 채워주고, 1인 부분은 그대로 두는 것입니다.

이후 softmax를 통해 normalize해줍니다.

```python
normalized_weights = F.softmax(weights, dim=-1)
# normalized_weights = [[1.0000, 0.0000, 0.0000],
#                       [0.5000, 0.5000, 0.0000],
#                       [0.3333, 0.3333, 0.3333]]
```

attention map을 잘 뽑아내는 것을 확인할 수 있습니다.

논문 리뷰에서 뒤를 보고 예측하면 반칙이라고 했죠?

casual mask는 이런식으로 구현됩니다.

뒤에 것을 참고하지 않고 앞에 것만 참고하도록 mask를 씌워줍니다.

또한 softmax를 통해 각 weight를 normalize해줍니다.

{:refdef: style="text-align: center;"}
![image](/assets/img/DT_code3.png)
{: refdef}

더 잘 이해해보기 위해,
query의 맨 위 파란색 줄을 가져왔습니다.

조그만 블럭 하나가 float값을 가지고 있으며, 새로줄 하나가 한 state, action, reward에 대한 정보를 가지고 있습니다.

이걸로 key에 대해 나랑 비슷한 놈들을 찾아보자!(행렬곱이 높은놈) 라고 하면,
(D,T)@(T,D)가 되어서 (D,D)가 됩니다.

그리고 이걸 softmax를 통해 normalize해주면 (D,D)가 나오게 됩니다.

즉, attention weight이 나오게 되겠죠?

```python
# attention (B, N, T, D)
attention = self.att_drop(normalized_weights @ v)
```

이걸 이후 value에 대해 곱해주면 (D,D)@(D,T)가 되어서 (D,T)가 됩니다.

즉, attention을 계산한 결과가 나오게 됩니다.

value를 한국말로 직역하는 과정에서 헷갈릴 수 있는데,

정말 가치! 를 말하는 것이 아닌, 해당 state, action, reward에 대한 정보를 말합니다.

~~(저는 처음 attention 공부할 때 많이 헷갈렸던 기억이 있네요 ㅎㅎ)~~

```python
# (B, N, T, D)
context = normalized_weights @ v
```

논문에서처럼 여러개의 attention을 쌓아서 사용해야 하는데,

그럼 원래 raw input과 dimension이 같아야겠죠?

원래 raw input dimension은 (Sequence len, attention_dim \* num_heads)이었고, 현재 나온 결과는 (Sequence len, attention_dim)이기 때문에,
raw input dimension과 같지 않습니다!

따라서 병렬 attention heads(여러 관점에서 본다고 했죠?)에서 계산했던 결과를 다시 합쳐줘야 합니다.

```python
# gather heads and project (B, N, T, D) -> (B, T, N*D)
attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)
```

이후 linear projection을 통해 마무리합니다.

```python
out = self.proj_drop(self.proj_net(attention))
```

자 이제 Attention 코드를 바탕으로 Block을 만들어봅시다.

지금 얼마나 했냐고요?

{:refdef: style="text-align: center;"}
![image](/assets/img/DT_code4.png)
{: refdef}

빨간박스만큼 했습니다. 힘내봅시다!

```python
class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x
```

forward를 보며 설명해보곘습니다.

```python
x = x + self.attention(x) # residual
```

input을 넣으면 위에서 한참 설명했던 attention을 진행합니다.

attention 결과물로 뭐가 나왔죠? (B, T, D)가 나왔습니다.

T는 sequence length, D는 dimensiond이었죠?

여기에 input을 더해주면 (B, T, D) + (B, T, D)가 되어서 (B, T, D)가 됩니다.

이것을 우리는 **residual connection**이라고 부릅니다.

이루 layer norm을 통해 정규화를 해줍니다.

그림에서는 add & norm이라고 표현되어 있습니다.

이후 Feed Forward Network를 통해 (B, T, D) -> (B, T, D)로 변환합니다.

또한 residual connection을 통해 (B, T, D) + (B, T, D)가 되어 (B, T, D)가 됩니다.

그림을 따라가면서 코드와 한번 다시 읽어보면서 이해해봅시다.

이해가 잘 될꺼예요 :)

{:refdef: style="text-align: center;"}
![image](/assets/img/DT_code5.png)
{: refdef}

이렇게 작성한 코드가 빨간색으로 표시된 Block입니다.

왼쪽에 Nx라고 되어있죠? 이렇게 Block을 여러번 통과시켜야 합니다.

다음 [post](https://daehwa00.github.io/rl/ml/2022/12/21/Decision-Transformer-code-review_2.html)에선 이렇게 만든 Block을 여러번 통과시키는 코드를 작성해보겠습니다!
