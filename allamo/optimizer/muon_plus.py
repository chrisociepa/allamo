"""
Muon+ optimizer: MomentUm Orthogonalized by Newton-Schulz with post-orthogonalization normalization.

Based on:
 - Muon (KellerJordan/Muon): https://github.com/KellerJordan/muon
 - Muon+ (arXiv:2602.21545): adds a normalization step after Newton-Schulz polar factorization.

Muon+ should only be used for hidden 2D weight matrices (Linear layers inside transformer blocks).
Embeddings, lm_head, biases, and normalization layers should be optimized with AdamW.
"""

import torch


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    Uses a quintic iteration whose coefficients maximize the slope at zero.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_plus_update(grad, momentum_buf, beta=0.95, ns_steps=5, nesterov=True, norm_mode='col'):
    """
    Compute a Muon+ update: momentum -> Newton-Schulz orthogonalization -> normalization.

    norm_mode controls the post-orthogonalization normalization:
      'col'  - column-wise L2 normalization
      'row'  - row-wise L2 normalization
      'joint' - Frobenius normalization (scale to unit norm)
      'none' - skip normalization (equivalent to vanilla Muon)
    """
    momentum_buf.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buf, beta) if nesterov else momentum_buf

    if update.ndim == 4:
        update = update.view(len(update), -1)

    update = zeropower_via_newtonschulz5(update, steps=ns_steps)

    if norm_mode == 'col':
        update = update / (update.norm(dim=0, keepdim=True) + 1e-7)
    elif norm_mode == 'row':
        update = update / (update.norm(dim=1, keepdim=True) + 1e-7)
    elif norm_mode == 'joint':
        update = update / (update.norm() + 1e-7)

    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


def _adam_update(grad, buf1, buf2, step, betas, eps):
    # TODO: replace with torch.optim.AdamW (fused=True) for better GPU throughput
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0] ** step)
    buf2c = buf2 / (1 - betas[1] ** step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonPlusWithAuxAdam(torch.optim.Optimizer):
    """
    Hybrid optimizer: Muon+ for hidden 2D weights, AdamW for everything else.

    Each param group must contain a `use_muon` flag.

    Muon+ groups accept: lr, weight_decay, momentum, ns_steps, norm_mode.
    AdamW  groups accept: lr, weight_decay, betas, eps.

    Example usage::

        muon_group = dict(
            params=hidden_2d_params, lr=0.02, weight_decay=0.01,
            momentum=0.95, ns_steps=5, norm_mode='col', use_muon=True,
        )
        adam_group = dict(
            params=other_params, lr=3e-4,
            betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0, use_muon=False,
        )
        optimizer = MuonPlusWithAuxAdam([muon_group, adam_group])
    """

    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group, "Each param group must specify use_muon=True/False"
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("ns_steps", 5)
                group.setdefault("norm_mode", "col")
                group.setdefault("weight_decay", 0.0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_plus_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group["momentum"],
                        ns_steps=group["ns_steps"],
                        norm_mode=group["norm_mode"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = _adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
