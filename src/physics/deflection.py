import torch


def gradient(psi, d):
    p = psi.squeeze(1) if psi.dim() == 4 else psi
    gy, gx = torch.gradient(p, spacing=(d, d), dim=(-2, -1))
    return gx, gy


def field_norm(fields, mask=None):
    v = torch.stack(fields, dim=1)
    if mask is not None:
        v = v * mask[:, None]
    return v.flatten(1).norm(dim=1)


def alpha_fields(pred_psi, true_psi, d):
    px, py = gradient(pred_psi, d)
    tx, ty = gradient(true_psi, d)
    return px - tx, py - ty, tx, ty


def alpha_rel_l2(pred_psi, true_psi, d, mask=None, eps=1e-6):
    ex, ey, tx, ty = alpha_fields(pred_psi, true_psi, d)
    return field_norm([ex, ey], mask) / (field_norm([tx, ty], mask) + eps)


def rel_l2(pred, target, mask=None, eps=1e-6):
    diff, ref = (pred - target).squeeze(1), target.squeeze(1)
    if mask is not None:
        diff, ref = diff * mask, ref * mask
    return diff.flatten(1).norm(dim=1) / (ref.flatten(1).norm(dim=1) + eps)