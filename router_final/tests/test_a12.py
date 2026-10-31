"""rfinal.a12.transpose_output: graf If sintetis -> keluaran (1, N, no) == transpose keluaran asli (ONNX Runtime)."""
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")


def _toy_if(path, no=6, n_a=10, n_b=4):
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(0)
    ya = rng.standard_normal((1, no, n_a)).astype(np.float32)
    yb = np.concatenate([rng.standard_normal((1, no, n_b)), np.zeros((1, no, n_a - n_b))], 2).astype(np.float32)
    vi = lambda n: helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, no, n_a])
    tg = helper.make_graph([helper.make_node("Constant", [], ["ya"], value=numpy_helper.from_array(ya))], "A", [], [vi("ya")])
    eg = helper.make_graph([helper.make_node("Constant", [], ["yb"], value=numpy_helper.from_array(yb))], "B", [], [vi("yb")])
    g = helper.make_graph([helper.make_node("Greater", ["mode", "c0"], ["c"]),
                           helper.make_node("Squeeze", ["c", "ax"], ["cond"]),
                           helper.make_node("If", ["cond"], ["y"], then_branch=tg, else_branch=eg)], "toy",
                          [helper.make_tensor_value_info("mode", TensorProto.FLOAT, [1])], [vi("y")],
                          initializer=[numpy_helper.from_array(np.zeros(1, np.float32), "c0"),
                                       numpy_helper.from_array(np.zeros(1, np.int64), "ax")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8
    onnx.save(m, str(path))
    return ya, yb


def test_transpose_output(tmp_path):
    from rfinal.a12 import transpose_output
    ya, yb = _toy_if(tmp_path / "a3.onnx")
    info = transpose_output(tmp_path / "a3.onnx", tmp_path / "a3t.onnx")
    assert info == dict(no=6, n=10)
    s0 = ort.InferenceSession(str(tmp_path / "a3.onnx"), providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(str(tmp_path / "a3t.onnx"), providers=["CPUExecutionProvider"])
    for mv, ref in ((1.0, ya), (-1.0, yb)):
        y0 = s0.run(None, {"mode": np.array([mv], np.float32)})[0]
        y1 = s1.run(None, {"mode": np.array([mv], np.float32)})[0]
        assert y1.shape == (1, 10, 6) and np.array_equal(y1, y0.transpose(0, 2, 1)) and np.array_equal(y0, ref)
    yt = s1.run(None, {"mode": np.array([-1.0], np.float32)})[0]
    assert np.all(yt[:, 4:] == 0) and np.any(yt[:, :4] != 0)          # awalan = anchor rute B, sisanya padding nol
