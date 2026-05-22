#
# Minimal, self-contained example of connection strength-based optimization on a toy
# two-task problem. Run:  python example.py
#
# It shows the only structural requirement of the method - a shared backbone with a
# task-specific BatchNorm after each conv - and the two-phase training loop.

import random
import torch
import torch.nn as nn

from conn_str import conn_str_optim

TASKS = ['task_a', 'task_b']


class STBlock(nn.Module):
    """ Shared conv + task-specific BatchNorm. `conv` and `bn[task]` are named so that
        the connection-strength optimizer can pair them. """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.ModuleDict({t: nn.BatchNorm2d(out_ch) for t in TASKS})
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, task):
        return self.relu(self.bn[task](self.conv(x)))


class ToyMTL(nn.Module):
    """ Shared backbone (with task-specific batch-norm) + a per-task head. """
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleList([STBlock(3, 16), STBlock(16, 16)])
        self.heads = nn.ModuleDict({t: nn.Conv2d(16, 1, kernel_size=1) for t in TASKS})

    def forward(self, x, task):
        h = x
        for block in self.backbone:
            h = block(h, task)
        return self.heads[task](h)


def per_task_grads(model, optimizer, criterion, x, y):
    """ Compute each task's gradient w.r.t. all parameters. """
    grad_dict = {}
    for t in TASKS:
        optimizer.zero_grad(set_to_none=True)
        criterion(model(x, task=t), y[t]).backward()
        grad_dict[t] = {n: (p.grad.clone() if p.grad is not None else None)
                        for n, p in model.named_parameters()}
    return grad_dict


def main():
    torch.manual_seed(0)
    random.seed(0)

    model = ToyMTL()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs, iters_per_epoch = 5, 10
    for epoch in range(num_epochs):
        for _ in range(iters_per_epoch):
            x = torch.randn(4, 3, 16, 16)
            y = {t: torch.randn(4, 1, 16, 16) for t in TASKS}

            # Mix the two phases: P >= e/E -> Phase 1, else Phase 2 (Algorithm 1).
            if random.random() >= epoch / num_epochs:
                # Phase 1 - learn task priority: update one task at a time.
                for t in TASKS:
                    optimizer.zero_grad(set_to_none=True)
                    criterion(model(x, task=t), y[t]).backward()
                    optimizer.step()
            else:
                # Phase 2 - conserve task priority via connection strength.
                grad_dict = per_task_grads(model, optimizer, criterion, x, y)
                optimizer.zero_grad(set_to_none=True)
                conn_str_optim(model, grad_dict, TASKS)
                optimizer.step()

        print(f"epoch {epoch + 1}/{num_epochs} done")

    print("OK - connection strength-based optimization ran successfully.")


if __name__ == "__main__":
    main()
