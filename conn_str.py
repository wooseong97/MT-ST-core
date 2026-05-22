#
# Connection strength-based optimization (core), from
# "Quantifying Task Priority for Multi-Task Optimization" (Jeong & Yoon, CVPR 2024).
#
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
#
# Self-contained: depends only on PyTorch. Drop it into any multi-task training loop
# whose shared backbone uses *task-specific batch-norm*, i.e. every shared conv named
# `<...>.conv` is followed by a per-task BatchNorm `<...>.bn[task]`.

import torch


def conn_str(conv_weight, bn_weight, bn_running_var, eps=1e-5):
    """Connection strength of each output channel to one task (Eq. 4 & 6).

    strength = (sum of squared conv weights per output channel)
               * (gamma^2 / (running_var + eps))   # task-specific batch-norm factor
    """
    strength = torch.sum(conv_weight.detach() ** 2, dim=(1, 2, 3))   # one value per out channel
    if bn_weight is not None:
        strength = strength * (bn_weight.detach() ** 2 / (bn_running_var.detach() + eps))
    return strength


def conn_str_grad(conv, bn_weight, bn_running_var, conv_grad, tasks):
    """Phase-2 gradient projection for a single shared conv.

    For every output channel, find its top-priority task from the (normalized)
    connection strength, then project each conflicting task gradient onto the plane
    orthogonal to the top-priority task's gradient. The projected gradients are summed
    into `conv.grad`.

    Returns True (and leaves `conv.grad` untouched) when the priority is degenerate -
    every channel shares one top-priority task - so the caller can sum the raw grads.
    """
    strength = {t: conn_str(conv, bn_weight[t], bn_running_var[t]) for t in tasks}
    proportion = {t: strength[t] / torch.sum(strength[t]) for t in tasks}     # Eq. 7
    pro_mat = torch.cat([torch.flatten(proportion[t]).unsqueeze(0) for t in tasks], 0)
    priority = torch.topk(pro_mat, len(tasks), dim=0)[1]    # row 0 = top-priority task per channel

    # Degenerate (uniform) priority: nothing to disentangle.
    if priority[0].unique().numel() == 1:
        return True

    grad_list = [torch.flatten(g, start_dim=1).clone() for g in conv_grad]   # [out, in*kh*kw] per task (copy: do not mutate caller's grads)
    for ti in range(len(tasks)):
        pos = (priority[0] == ti)        # channels whose top-priority task is ti
        for tj in range(len(tasks)):
            if ti == tj:
                continue
            gi = torch.flatten(grad_list[ti][pos])
            gj = torch.flatten(grad_list[tj][pos])
            dot = torch.dot(gi, gj)
            if dot < 0:                  # conflict: project tj off ti
                gj = gj - dot / (torch.norm(gi) ** 2) * gi
                grad_list[tj][pos] = gj.reshape(grad_list[tj][pos].shape)

    for g in grad_list:
        g = g.reshape(conv.shape).clone().detach()
        conv.grad = g if conv.grad is None else conv.grad + g
    return False


@torch.no_grad()
def conn_str_optim(model, grad_dict, tasks):
    """Combine per-task gradients into `param.grad`, using the Phase-2 projection.

    Args:
        model:     the multi-task model (with task-specific batch-norm in the backbone).
        grad_dict: grad_dict[task][name] = task `task`'s gradient for parameter `name`
                   (None where that task does not touch the parameter).
        tasks:     list of task names.

    A conv `<X>.conv.weight` is treated as a connection-strength layer iff a per-task
    BatchNorm `<X>.bn[task]` exists for every task (matched by replacing `conv`->`bn`
    in the name). Every other parameter just accumulates the per-task gradients.

    Call `optimizer.zero_grad()` before this so gradients start from zero, then
    `optimizer.step()` afterwards.
    """
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    def accumulate(param, name):
        for t in tasks:
            g = grad_dict[t][name]
            if g is None:
                continue
            param.grad = g if param.grad is None else param.grad + g

    for name, param in model.named_parameters():
        bn_w = {t: name.replace('conv', 'bn').replace('.weight', '.%s.weight' % t) for t in tasks}
        bn_v = {t: name.replace('conv', 'bn').replace('.weight', '.%s.running_var' % t) for t in tasks}
        is_conn_layer = (name.endswith('.weight') and 'conv' in name
                         and all(bn_w[t] in params for t in tasks)
                         and all(bn_v[t] in buffers for t in tasks)
                         and all(grad_dict[t][name] is not None for t in tasks))
        if is_conn_layer:
            bn_weight = {t: params[bn_w[t]] for t in tasks}
            bn_running_var = {t: buffers[bn_v[t]] for t in tasks}
            conv_grad = [grad_dict[t][name] for t in tasks]
            skipped = conn_str_grad(param, bn_weight, bn_running_var, conv_grad, tasks)
            if skipped:
                accumulate(param, name)
        else:
            accumulate(param, name)
