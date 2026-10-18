"""
TensorRT untuk engine A3 (router_final). Buffer memakai tensor CUDA PyTorch (tanpa pycuda).
Diporting dari job_main eksperimen/tahap4_final.py (build, Eng). Dipakai scripts/evaluate.py --backend trt.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch


def build_engine(onnx_path, engine_path, workspace_gb=1, fp16_flag=False):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)    # presisi = tipe di ONNX
    net = builder.create_network(flags)
    parser = trt.OnnxParser(net, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16_flag and not flags and hasattr(trt.BuilderFlag, "FP16"):     # TensorRT < 10 (tanpa STRONGLY_TYPED)
        cfg.set_flag(trt.BuilderFlag.FP16)
    t = time.perf_counter()
    eb = builder.build_serialized_network(net, cfg)
    assert eb is not None, f"gagal build {onnx_path}"
    Path(engine_path).write_bytes(eb)
    return dict(engine=str(engine_path), build_s=time.perf_counter() - t, trt=trt.__version__,
                strongly_typed=bool(flags))


class A3Engine:
    """infer(img (1,3,H,W) FP32 [0,1] CPU/GPU, thr, mode=-1) -> (y CUDA (1,4+nc,N_A), use_a bool, score float)."""

    def __init__(self, engine_path, device="cuda:0"):
        import tensorrt as trt
        self.dev = torch.device(device)
        t2t = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
               trt.int64: torch.int64, trt.bool: torch.bool}
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.buf = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(n))
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.ctx.set_input_shape(n, shape)
            self.buf[n] = torch.zeros(shape, dtype=t2t[self.engine.get_tensor_dtype(n)], device=self.dev)
            self.ctx.set_tensor_address(n, self.buf[n].data_ptr())
        assert {"image", "mode", "thr", "y", "use_a", "score"} <= set(self.buf), sorted(self.buf)
        self.stream = torch.cuda.Stream(self.dev)
        self.h_use, self.h_score = torch.zeros(1).pin_memory(), torch.zeros(1).pin_memory()

    @torch.no_grad()
    def infer(self, img, thr, mode=-1.0):
        # Stream engine menunggu stream PyTorch: `img` (dihitung di stream default) harus selesai, dan clone()
        # keluaran panggilan sebelumnya tidak boleh tertimpa sebelum selesai dibaca.
        self.stream.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(self.stream):
            self.buf["image"].copy_(img, non_blocking=True)
            self.buf["thr"].fill_(float(thr)); self.buf["mode"].fill_(float(mode))
            assert self.ctx.execute_async_v3(self.stream.cuda_stream), "execute_async_v3 gagal"
            self.h_use.copy_(self.buf["use_a"], non_blocking=True)
            self.h_score.copy_(self.buf["score"], non_blocking=True)
        self.stream.synchronize()
        return self.buf["y"], bool(float(self.h_use[0]) > 0.5), float(self.h_score[0])
