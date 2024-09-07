# Switch Transformers

PyTorch implementation of the [Switch Transformer paper](https://arxiv.org/abs/2101.03961).
Read also my [blogpost](https://srishti-git1110.github.io/blog/moes/) covering the paper.

![Switch Layer](switch_layer.png#center)

## News
- Now supporting the latest aux_loss free load balancing technique from [this](https://arxiv.org/pdf/2408.15664v1) paper. Simply pass `use_biased_gating=True` while instantiating the `SwitchTransformer` class. 

Rest all is taken care of!

```python
switch_transformer = SwitchTransformer(
    inp_dim,
    num_experts,
    num_heads,
    vocab_size,
    use_biased_gating=True,
).cuda()
```

## Usage

1. Clone the repo

```
git clone https://github.com/srishti-git1110/torch-switch-transformers.git
```

2. Navigate to the correct directory

```
cd torch-switch-transformers
```

3. Install the required dependencies

```
pip install -r requirements.txt
```

4. Usage

#### With aux_loss

```python
import torch

from switch_transformers import SwitchTransformer

inp_dim = 512
num_experts = 8
num_heads = 8
vocab_size = 50000

switch_transformer = SwitchTransformer(
    inp_dim,
    num_experts,
    num_heads,
    vocab_size,
    use_aux_loss=True, # optional since this is used by default if use_biased_gating is not True
).cuda()

x = torch.randn(2, 1024, inp_dim).cuda()
output, total_aux_loss = switch_transformer(x)
```

#### With aux_loss free load balancing

```python
switch_transformer = SwitchTransformer(
    inp_dim,
    num_experts,
    num_heads,
    vocab_size,
    use_biased_gating=True,
).cuda()
```