import torch


def gradient(psi, d):
    p = psi.squeeze(1) if psi.dim() == 4 else psi
    gy, gx = torch.gradient(p, spacing=(d, d), dim=(-2, -1))
    return gx, gy


def alpha_rel_l2(pred_psi, true_psi, d, mask=None, eps=1e-6):
    px, py = gradient(pred_psi, d)
    tx, ty = gradient(true_psi, d)
    err, ref = (px - tx) ** 2 + (py - ty) ** 2, tx**2 + ty**2
    if mask is not None:
        err, ref = err * mask, ref * mask
    return err.sum((-2, -1)).sqrt() / (ref.sum((-2, -1)).sqrt() + eps)


def rel_l2(pred, target, mask=None, eps=1e-6):
    diff, ref = (pred - target).squeeze(1), target.squeeze(1)
    if mask is not None:
        diff, ref = diff * mask, ref * mask
    return diff.flatten(1).norm(dim=1) / (ref.flatten(1).norm(dim=1) + eps)