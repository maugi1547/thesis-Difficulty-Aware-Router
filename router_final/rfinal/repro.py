"""Sidik jari bobot untuk uji reproduksi (tidak bergantung metadata pickle seperti tanggal/argv)."""
from __future__ import annotations

import hashlib

import torch


def state_hash(obj):
    """sha256 atas (nama, dtype, shape, bytes) semua tensor state_dict, urut nama."""
    sd = obj.state_dict() if hasattr(obj, "state_dict") else obj
    h = hashlib.sha256()
    for k in sorted(sd):
        v = sd[k]
        if not torch.is_tensor(v):
            continue
        t = v.detach().cpu().contiguous()
        h.update(k.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    return h.hexdigest()


def max_abs_diff(a, b):
    """Selisih absolut maksimum per tensor antara dua state_dict (untuk diagnosis bila hash berbeda)."""
    sa = a.state_dict() if hasattr(a, "state_dict") else a
    sb = b.state_dict() if hasattr(b, "state_dict") else b
    assert sa.keys() == sb.keys(), "kunci state_dict berbeda"
    out = {}
    for k in sa:
        if torch.is_tensor(sa[k]) and sa[k].is_floating_point():
            out[k] = float((sa[k].float() - sb[k].float()).abs().max()) if sa[k].numel() else 0.0
    return out
