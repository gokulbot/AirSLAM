#!/usr/bin/env python3
"""
Make TensorRT-8-era ONNX models parse cleanly on TensorRT 10.

Why: TRT 10 enforces matching integer types on type-symmetric ops (elementwise, Concat,
Where, comparisons) and natively uses INT64 for shape math. AirSLAM's pre-exported ONNX
mix int64 (Shape/Unsqueeze outputs) with int32 constants — TRT 8 silently coerced these,
TRT 10 rejects them ("incompatible input types Int64 and Int32").

Fix: at each mismatched op, cast the int32 inputs UP to int64 (aligning with TRT 10's
internal int64 shape handling). This preserves the model's exact I/O contract and works
for dynamic-shape models, so no static-freezing or C++ changes are needed.

Usage:
    python3 fix_int_types.py [--backup=DIR] model1.onnx [model2.onnx ...]
"""
import sys, os, copy, shutil
import onnx
from onnx import TensorProto, helper

I64, I32 = TensorProto.INT64, TensorProto.INT32
MATCH = {"Add", "Sub", "Mul", "Div", "Mod", "Pow", "Min", "Max", "Sum", "Mean",
         "Concat", "Where", "Equal", "Greater", "Less", "GreaterOrEqual",
         "LessOrEqual", "And", "Or", "Xor"}
BOOL_OUT = {"Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual", "And", "Or", "Xor"}


def fix(path, backup_dir=None):
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        b = os.path.join(backup_dir, os.path.basename(path))
        if not os.path.exists(b):
            shutil.copy2(path, b)

    m = onnx.load(path)
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception as e:
        print("  (shape inference warning:", e, ")")
    g = m.graph

    et = {}
    for vi in list(g.input) + list(g.output) + list(g.value_info):
        et[vi.name] = vi.type.tensor_type.elem_type
    for init in g.initializer:
        et[init.name] = init.data_type

    newnodes, n = [], 0
    for node in g.node:
        if node.op_type in MATCH:
            types = [et.get(i, 0) for i in node.input]
            if I32 in types and I64 in types:
                nc = copy.deepcopy(node)
                for k, i in enumerate(list(nc.input)):
                    if et.get(i, 0) == I32:
                        outn = f"{i}_toi64_{n}"
                        newnodes.append(helper.make_node("Cast", [i], [outn], to=I64, name=f"CastUp_{n}"))
                        nc.input[k] = outn
                        et[outn] = I64
                        n += 1
                if node.op_type not in BOOL_OUT:
                    for o in nc.output:
                        et[o] = I64
                newnodes.append(nc)
                continue
        newnodes.append(copy.deepcopy(node))

    del g.node[:]
    g.node.extend(newnodes)
    onnx.save(m, path)
    print(f"  {os.path.basename(path)}: inserted {n} int32->int64 cast(s)")


if __name__ == "__main__":
    files = [a for a in sys.argv[1:] if not a.startswith("--")]
    bdir = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--backup=")), None)
    if not files:
        print(__doc__)
        sys.exit(1)
    for p in files:
        fix(p, bdir)
