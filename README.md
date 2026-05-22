# Connection Strength-based Optimization (core)

A small, self-contained implementation of the optimizer from

> **Quantifying Task Priority for Multi-Task Optimization**
> Wooseong Jeong, Kuk-Jin Yoon (KAIST), *CVPR 2024*
> [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Jeong_Quantifying_Task_Priority_for_Multi-Task_Optimization_CVPR_2024_paper.html) · [arXiv:2406.02996](https://arxiv.org/abs/2406.02996)

This repo contains **only the method** ([`conn_str.py`](conn_str.py), depends on PyTorch
alone) plus a runnable toy example. Drop it into your own multi-task training loop. For
the full experiments (NYUD-v2 / PASCAL-Context, MTI-Net, baselines), see the main repo.

## Idea

Each task's gradient on a *shared* parameter has a different importance — its **task
priority**. We quantify it with the **connection strength** of every shared convolution
output channel to each task, measured from the conv weights scaled by that task's
batch-norm (Eq. 4, 6, 7 in the paper). Optimization runs in two phases:

- **Phase 1 — learn task priority:** update one task's gradient at a time.
- **Phase 2 — conserve task priority:** for each output channel, find its top-priority
  task from the connection strength and project conflicting task gradients onto the plane
  orthogonal to the top-priority gradient before summing.

The two phases are mixed stochastically: at epoch `e` of `E`, draw `P ~ U(0,1)` and run
Phase 1 if `P >= e/E`, else Phase 2 (so Phase 2 dominates late in training).

## Requirement

The method needs a shared backbone with **task-specific batch-norm**: each shared conv
`<X>.conv` must be followed by a per-task BatchNorm `<X>.bn[task]` (so the names pair by
replacing `conv` → `bn`). `conn_str_optim` automatically applies the projection to those
conv/BN pairs and just sums the per-task gradients for every other parameter.

## API

```python
from conn_str import conn_str_optim

# grad_dict[task][param_name] = that task's gradient for the parameter (None if untouched)
optimizer.zero_grad()
conn_str_optim(model, grad_dict, tasks)   # Phase-2 update; fills param.grad
optimizer.step()
```

## Integration (Phase 1 / Phase 2 mixing)

```python
import random

for epoch in range(num_epochs):
    for x, y in loader:                       # y: {task: target}
        if random.random() >= epoch / num_epochs:
            # Phase 1: update one task at a time
            for t in tasks:
                optimizer.zero_grad()
                criterion(model(x, task=t), y[t]).backward()
                optimizer.step()
        else:
            # Phase 2: connection strength
            grad_dict = {}
            for t in tasks:
                optimizer.zero_grad()
                criterion(model(x, task=t), y[t]).backward()
                grad_dict[t] = {n: (p.grad.clone() if p.grad is not None else None)
                                for n, p in model.named_parameters()}
            optimizer.zero_grad()
            conn_str_optim(model, grad_dict, tasks)
            optimizer.step()
```

See [`example.py`](example.py) for a complete, runnable version on a toy two-task problem.

## Run the example

```bash
pip install torch
python example.py
```

## Citation

```bibtex
@inproceedings{jeong2024quantifying,
  title     = {Quantifying Task Priority for Multi-Task Optimization},
  author    = {Jeong, Wooseong and Yoon, Kuk-Jin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {363--372},
  year      = {2024}
}
```

## License

Released under **CC BY-NC 4.0** (non-commercial). See [LICENSE](LICENSE).
