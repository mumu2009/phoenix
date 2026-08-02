import json
import onnx
import numpy as np
from onnx import numpy_helper


def attr_dict(node):
    d = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INTS:
            d[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.INT:
            d[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOATS:
            d[a.name] = list(a.floats)
        elif a.type == onnx.AttributeProto.FLOAT:
            d[a.name] = a.f
        else:
            d[a.name] = str(a)
    return d


def inspect(path, name):
    m = onnx.load(path)
    init = {i.name: i for i in m.graph.initializer}
    print(f"\n=== {name} ===")
    print(f"inputs: {[(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in m.graph.input]}")
    print(f"outputs: {[(v.name, [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in m.graph.output]}")
    for n in m.graph.node:
        attrs = attr_dict(n)
        # get weight shapes
        ws = []
        for inp in n.input:
            if inp in init:
                arr = numpy_helper.to_array(init[inp])
                ws.append(arr.shape)
        print(n.op_type, [i for i in n.input[:2]], [o for o in n.output], attrs, ws)


inspect(r'D:\_phoenix\_079\v6.0Alixander\phoenix\runtime_store\models\ijepa\speech_16k\model_encoder.onnx', 'encoder')
inspect(r'D:\_phoenix\_079\v6.0Alixander\phoenix\runtime_store\models\ijepa\speech_16k\model_decoder.onnx', 'decoder')
