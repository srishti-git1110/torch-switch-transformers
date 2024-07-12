# Switch Transformers

PyTorch implementation of the [Switch Transformer paper](https://arxiv.org/abs/2101.03961).
Read also my [blogpost](https://srishti-git1110.github.io/blog/moes/) covering the paper.

![Switch Layer](switch_layer.png#center)

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
    vocab_size
).cuda()

x = torch.randn(2, 1024, inp_dim).cuda()
output, total_aux_loss = switch_transformer(x)
```