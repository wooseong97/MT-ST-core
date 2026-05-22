# Quantifying Task Priority for Multi-Task Optimization [CVPR 2024]

**Official PyTorch implementation (method core) of [*Quantifying Task Priority for Multi-Task Optimization*](https://openaccess.thecvf.com/content/CVPR2024/html/Jeong_Quantifying_Task_Priority_for_Multi-Task_Optimization_CVPR_2024_paper.html) [CVPR 2024].**

Wooseong Jeong & Kuk-Jin Yoon, Korea Advanced Institute of Science and Technology (KAIST)

The goal of multi-task learning is to learn diverse tasks within a single unified network. As each task has its own unique objective function, conflicts emerge during training, resulting in negative transfer among them. Earlier research identified these conflicting gradients in shared parameters between tasks and attempted to realign them in the same direction. However, we prove that such optimization strategies lead to sub-optimal Pareto solutions due to their inability to accurately determine the individual contributions of each parameter across various tasks. In this paper, we propose the concept of task priority to evaluate parameter contributions across different tasks. To learn task priority, we identify the type of connections related to links between parameters influenced by task-specific losses during backpropagation. The strength of connections is gauged by the magnitude of parameters to determine task priority. Based on these, we present a new method named connection strength-based optimization which consists of two phases. The first phase learns the task priority within the network, while the second phase modifies the gradients while upholding this priority. This ultimately leads to finding new Pareto optimal solutions for multiple tasks. Through extensive experiments, we show that our approach greatly enhances multi-task performance in comparison to earlier gradient manipulation methods.

---

## Overview

This repository provides the **core of the method** — the connection strength-based optimizer in [`conn_str.py`](conn_str.py) (depends on PyTorch alone) together with a runnable toy example. You can drop it into your own multi-task training loop. For the full experiments (NYUD-v2 / PASCAL-Context, MTI-Net, gradient-manipulation and loss-scaling baselines), see the main repository.

```
.
├── conn_str.py    # the method: connection strength (Eq. 4/6/7) + Phase-2 gradient projection
├── example.py     # minimal runnable toy two-task example
├── README.md
└── LICENSE
```

## Method

Each task's gradient on a *shared* parameter has a different importance — its **task priority**. We quantify it with the **connection strength** of every shared convolution output channel to each task, measured from the conv weights scaled by that task's batch-norm (Eq. 4, 6, 7). Optimization runs in two phases:

- **Phase 1 — learn task priority:** update one task's gradient at a time.
- **Phase 2 — conserve task priority:** for each output channel, find its top-priority task from the connection strength and project conflicting task gradients onto the plane orthogonal to the top-priority gradient before summing.

The two phases are mixed stochastically: at epoch `e` of `E`, draw `P ~ U(0,1)` and run Phase 1 if `P >= e/E`, else Phase 2 (so Phase 2 dominates late in training).

## Requirement

The method needs a shared backbone with **task-specific batch-norm**: each shared conv `<X>.conv` must be followed by a per-task BatchNorm `<X>.bn[task]` (so the names pair by replacing `conv` → `bn`). `conn_str_optim` automatically applies the projection to those conv/BN pairs and simply sums the per-task gradients for every other parameter.

## Usage

```python
from conn_str import conn_str_optim

# grad_dict[task][param_name] = that task's gradient for the parameter (None if untouched)
optimizer.zero_grad()
conn_str_optim(model, grad_dict, tasks)   # Phase-2 update; fills param.grad
optimizer.step()
```

Two-phase training loop (Phase 1 / Phase 2 mixing):

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

## Run the example

```bash
pip install torch
python example.py
```

`example.py` is a complete, runnable version on a toy two-task problem.

## Contact

Wooseong Jeong: stk14570@kaist.ac.kr

## Reference

If you find this code useful, please cite the following paper:

```bibtex
@inproceedings{jeong2024quantifying,
  title={Quantifying Task Priority for Multi-Task Optimization},
  author={Jeong, Wooseong and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={363--372},
  year={2024}
}
```

## Acknowledgement

The full implementation builds on the open-source [Multi-Task-Learning-PyTorch (MTI-Net)](https://github.com/SimonVandenhende/Multi-Task-Learning-PyTorch) and [ASTMT](https://github.com/facebookresearch/astmt) projects. We thank the authors for releasing their code.
