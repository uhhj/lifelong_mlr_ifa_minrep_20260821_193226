"""Append-only reference span. No centering, whitening, PCA or learned state."""
import torch


def append_basis(U, reference):
    r = reference.detach().cpu().double().flatten()
    assert r.numel() == 512 and torch.isfinite(r).all() and r.norm() > 0
    r = r / r.norm()
    v = r.clone()
    for _ in range(2):
        v = v - U @ (U.T @ v)
    residual = float(v.norm())
    if residual > 1e-7:
        U = torch.cat((U, (v/v.norm())[:, None]), dim=1)
    return U, residual


def validate(U, refs):
    R = torch.stack(refs).double()
    R = R/R.norm(dim=1, keepdim=True)
    errors = {}
    for dtype, tol_orth, tol_ref in [(torch.float64, 1e-10, 1e-6), (torch.float32, 1e-5, 1e-5)]:
        u, r = U.to(dtype), R.to(dtype)
        orth = float((u.T@u-torch.eye(u.shape[1],dtype=dtype)).norm())
        recon = float((r-(r@u)@u.T).norm(dim=1).max())
        errors[str(dtype)] = dict(orthogonality_fro=orth, max_unit_reference_error=recon)
        assert orth <= tol_orth and recon <= tol_ref, errors
    return errors


def project(z, U):
    assert z.dtype == U.dtype == torch.float32 and not U.requires_grad
    return (z@U)@U.T


def angle_hinge(z, pos, neg, margin):
    # Two native normalizations retained, for engineering checks only.
    def n(x): return x/(x.norm(dim=-1,keepdim=True)+1e-8)
    z, pos, neg = n(n(z)), n(n(pos)), n(n(neg))
    a = torch.acos((z*pos).sum(-1).clamp(-1+1e-6,1-1e-6))
    b = torch.acos((z*neg).sum(-1).clamp(-1+1e-6,1-1e-6))
    return torch.relu(a-b+margin), a, b
