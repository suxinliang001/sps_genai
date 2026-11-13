# Assignment 4

## Part 1. Model deployment

Both models were trained on the 50,000-image CIFAR-10 training split. The FastAPI app serves the five-epoch Diffusion model at `POST /diffusion/generate` and the eight-epoch Energy-Based Model at `POST /ebm/generate`. The repository contains the checkpoints and instructions for running the endpoints. [GitHub repository](https://github.com/suxinliang001/sps_genai)

## Part 2. Building blocks of a diffusion model

### 1. Sinusoidal timestep embedding

Let $d$ be the embedding dimension, $h = \lfloor d/2 \rfloor$, and $M$ the maximum period. Use zero-based indexing. For each $j = 0, \ldots, h-1$, compute a frequency $\omega_j = \exp(-\log(M)j/h) = M^{-j/h}$ and an angle $t\omega_j$ in radians. Using the cosine-first layout in the Assignment 4 implementation, the components are

    eᵢ(t) = cos(t M^(−i/h))          for 0 ≤ i < h
    eᵢ(t) = sin(t M^(−(i−h)/h))      for h ≤ i < 2h

If $d$ is odd, append a zero at index $2h$. Each frequency contributes one cosine and one sine. The frequencies are fixed, and the timestep changes the phase of each pair. [Reference implementation](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/nn.py)

### 2. Eight-dimensional example at t = 1

With $d = 8$, $h = 4$, and $M = 10000 = 10^4$, the frequencies are $\omega_j = (10^4)^{-j/4} = 10^{-j}$. Multiplying by $t = 1$ gives the angles $1$, $0.1$, $0.01$, and $0.001$ radians. Taking the four cosines followed by the four sines gives

    [0.54030231, 0.99500417, 0.99995000, 0.99999950,
     0.84147098, 0.09983342, 0.00999983, 0.00100000]

The following code computes the vector to eight decimal places.

```python
import math
import torch

j = torch.arange(4, dtype=torch.float64)
angles = 1.0 * torch.exp(-math.log(10_000) * j / 4)
embedding = torch.cat((angles.cos(), angles.sin()))
print("[" + ", ".join(f"{value:.8f}" for value in embedding.tolist()) + "]")
```

### 3. Transformer positional encoding and diffusion time embedding

Both encode an index with sine and cosine pairs at several fixed frequencies. In the original Transformer, the index is a token's position in a sequence. The model adds a positional vector to each token embedding so attention has access to token order. [Attention Is All You Need, section 3.5](https://arxiv.org/abs/1706.03762)

In a diffusion U-Net, the index is the current diffusion step, which determines the noise level. A single timestep vector conditions the residual blocks for the entire noisy image. Token positions can vary within one sequence, while the diffusion step is shared by all spatial locations of one image.

### 4. Bottleneck resolution

For a stride-2 convolution with kernel size 3, padding 1, and dilation 1, each spatial dimension $n$ becomes $\lfloor(n + 2 - 3)/2\rfloor + 1$. For even $n$, this equals $n/2$. Applying the formula to height and width gives

    64 × 64 → 32 × 32 → 16 × 16 → 8 × 8

The bottleneck resolution is **8 × 8**, assuming each block uses padding that halves the spatial dimensions. Channel counts are independent of this calculation.

```python
import torch
from torch import nn

image = torch.zeros(1, 3, 64, 64)
downsample = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
for _ in range(3):
    image = downsample(image)
    print(tuple(image.shape[-2:]))  # (32, 32), (16, 16), (8, 8)
```

### 5. U-Net output and the noise-prediction loss

Let $x_0$ be a clean image and $\beta_s$ the variance at step $s$. Define $\alpha_s = 1 - \beta_s$. The cumulative product ᾱₜ is the product of $\alpha_1$ through $\alpha_t$. Draw $t$ uniformly from the training steps and independently draw standard Gaussian noise $\epsilon$ with the same shape as $x_0$. The forward process gives

    xₜ = √(ᾱₜ) x₀ + √(1 − ᾱₜ) ε.

The U-Net receives $(x_t, t)$ and returns $\epsilon_\theta(x_t,t)$, a prediction of the noise with the same shape as $x_t$. The target is the exact noise tensor $\epsilon$ used to construct $x_t$. For a batch with $B$ images and $N = C H W$ entries per image, the simplified DDPM loss is

    L = (1 / BN) Σ_b Σ_k (ε̂[b,k] − ε[b,k])².

Backpropagating this mean squared error updates $\theta$. During generation, the predicted noise is used to move to a less noisy image at the next reverse step. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

```python
import torch
from torch.nn import functional as F
from app.diffusion import DiffusionProcess, DiffusionUNet

model = DiffusionUNet()
process = DiffusionProcess()
x0 = torch.rand(2, 3, 32, 32) * 2 - 1
t = torch.tensor([249, 749])  # Zero-based schedule indices.
epsilon = torch.randn_like(x0)
alpha_bar = process.alpha_bar[t, None, None, None]
xt = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * epsilon
predicted_noise = model(xt, t)
loss = F.mse_loss(predicted_noise, epsilon)
loss.backward()
print(tuple(predicted_noise.shape))  # (2, 3, 32, 32)
```

## Part 3. Building blocks of an energy model

### 6. Basic gradient calculations

**(a)** The power rule gives $d(x^2)/dx = 2x$ and the linear term gives $d(3x)/dx = 3$. Therefore $dy/dx = 2x + 3$. At $x = 2$, this is $2(2) + 3 = 7$. The call to `backward()` stores this gradient on the leaf tensor `x`.

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3 * x
y.backward()
print("x.grad =", x.grad)  # tensor([7.])
```

**(b)** With `requires_grad=False`, the forward computation still gives $y = 10$. There is no gradient-tracked input, so `y.requires_grad` is false and `y.grad_fn` is `None`. Calling `y.backward()` raises a `RuntimeError`. The field `x.grad` stays `None`.

```python
import torch

x = torch.tensor([2.0], requires_grad=False)
y = x**2 + 3 * x
print(y.item(), y.requires_grad, y.grad_fn)  # 10.0 False None
try:
    y.backward()
except RuntimeError as error:
    print(type(error).__name__, x.grad)  # RuntimeError None
```

**(c)** The default is `requires_grad=False`, so a new tensor made with `torch.tensor(...)` does not track gradients unless requested. [PyTorch tensor documentation](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html)

```python
import torch

print(torch.tensor([2.0]).requires_grad)  # False
```

### 7. Introduce weights

**(a)** For $y = w_0 x^2 + w_1 x$, holding the weights fixed gives $\partial y/\partial x = 2w_0 x + w_1 = 2(1)(2) + 3 = 7$. The supplied code creates $w$ with the default gradient flag, so the result is `w.grad = None`.

```python
import torch

x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([1.0, 3.0])
y = w[0] * x**2 + w[1] * x
y.backward()
print("x.grad =", x.grad)  # tensor([7.])
print("w.grad =", w.grad)  # None
```

**(b)** Holding $x$ fixed, the coefficient of $w_0$ in $y$ is $x^2$, and the coefficient of $w_1$ is $x$. At $x = 2$, $\partial y/\partial w_0 = x^2 = 4$ and $\partial y/\partial w_1 = x = 2$. Enable gradients on $w$. Since this calculation only needs weight gradients, $x$ can use the default flag.

```python
import torch

x = torch.tensor([2.0])
w = torch.tensor([1.0, 3.0], requires_grad=True)
y = w[0] * x**2 + w[1] * x
y.backward()
print("w.grad =", w.grad)  # tensor([4., 2.])
```

**(c)** The default is `requires_grad=False`. The new tensor can participate as a constant in an expression whose other operands require gradients. In part (a), the graph still records the operations needed to compute $\partial y/\partial x$. [PyTorch tensor documentation](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html)

### 8. Breaking the graph

The operation `y.detach()` gives $z$ the same value as $y$ but removes its autograd connection to $x$. The multiplication `w = z * 2` has no gradient-tracked input, so `w.backward()` raises a `RuntimeError`. Enabling gradients on a detached $z$ would start a new graph at $z$.

Use `y.clone()` to keep $z$ as a separate tensor and preserve the connection to $x$. The chain rule gives $dw/dx = (dw/dz)(dz/dy)(dy/dx) = 2 \cdot 1 \cdot 3 = 6$. [PyTorch detach documentation](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html)

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
y = x * 3
z = y.clone()
w = z * 2
w.backward()
print("x.grad =", x.grad)  # tensor([6.])
```

### 9. Gradient accumulation

The first backward call adds $dy_1/dx = 2$ to `x.grad`. The second call adds $dy_2/dx = 3$ to the existing value, giving $2 + 3 = 5$. Reset `x.grad` to `None` before the second backward call to obtain its gradient by itself. When training a model, call `optimizer.zero_grad(set_to_none=True)` before the next independent backward pass.

```python
import torch

x = torch.tensor([1.0], requires_grad=True)
(x * 2).backward()
print("First backward:", x.grad)  # tensor([2.])
(x * 3).backward()
print("Accumulated:", x.grad)     # tensor([5.])
x.grad = None
(x * 3).backward()
print("After resetting:", x.grad) # tensor([3.])
```
