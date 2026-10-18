from pathlib import Path
def apply_source_patch(path, old, new, done_marker, name):
    """
    Ganti `old` -> `new` tepat satu kali di file `path`, mempertahankan EOL file (LF/CRLF).
    Idempotent: bila `done_marker` sudah ada, file tidak diubah. Returns status string.
    """
    path = Path(path)
    raw = path.read_bytes().decode("utf-8")
    if done_marker in raw:
        return f"ℹ️ {name}: sudah ada — tidak di-patch."
    if "\r\n" in raw:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    n = raw.count(old)
    if n != 1:
        raise RuntimeError(f"{name}: pola ditemukan {n}x (harus 1) di {path} — versi file berbeda, cek manual.")
    path.write_bytes(raw.replace(old, new).encode("utf-8"))
    return f"✅ {name}: di-patch."


PATCH_P2 = dict(
    name="P2 keputusan & logit gate per-gambar (block.py, eval)",
    rel="ultralytics/nn/modules/block.py",
    done_marker="self._last_gate_logit_raw = raw_logit_diff.detach()",
    old=("            if not torch.jit.is_tracing():\n"
         "                self.current_activation_prob = gate_mask.mean().detach()\n\n"
         "            f_c2f = self.compute_expert(f_p3, f_p2_back)\n"),
    new=("            if not torch.jit.is_tracing():\n"
         "                self.current_activation_prob = gate_mask.mean().detach()\n"
         "                # [TAHAP0] per-gambar: keputusan (uji konsistensi vs compute_gate_only) dan logit\n"
         "                # mentah sebelum kalibrasi (sebaran logit di log validasi)\n"
         "                self._last_gate_decision = gate_mask.view(B).detach()\n"
         "                self._last_gate_logit_raw = raw_logit_diff.detach()\n\n"
         "            f_c2f = self.compute_expert(f_p3, f_p2_back)\n"),
)

PATCH_CONF_SIGN = dict(
    name="Tanda conf di difficulty score (loss.py): - menjadi +",
    rel="ultralytics/utils/loss.py",
    done_marker="diff_score = entropy_f + var_f + conf_f",
    old=("                # TODO: verifikasi -- router.last_conf adalah UNCERTAINTY (1 - prob, tinggi =\n"
         "                # tidak yakin), sama arah dengan entropy dan var. Mengurangkannya di sini\n"
         "                # berarti ketidakpastian kelas yang lebih tinggi MENURUNKAN skor kesulitan.\n"
         "                # Apakah tanda minus ini disengaja (mis. dulu conf berarti keyakinan)?\n"
         "                diff_score = entropy_f + var_f - conf_f\n"),
    new=("                # [FIX 2026-10-09] router.last_conf adalah UNCERTAINTY (1 - prob, tinggi = tidak\n"
         "                # yakin), searah dengan entropy dan var. Tanda minus versi lama adalah salah ketik\n"
         "                # (dikonfirmasi penulis): ketiga sinyal ketidakpastian kini DIJUMLAHKAN.\n"
         "                # Run sebelum fix ini memakai entropy + var - conf (tidak sebanding langsung).\n"
         "                diff_score = entropy_f + var_f + conf_f\n"),
)



import sys
root = Path(sys.argv[1])
for P_ in (PATCH_P2, PATCH_CONF_SIGN):
    print(apply_source_patch(root / P_["rel"], P_["old"], P_["new"], P_["done_marker"], P_["name"]))
