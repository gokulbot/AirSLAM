from ultralytics import YOLO
import onnx, sys
m = YOLO("yolov8m.pt")                      # detection (boxes), not -seg
p = m.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False)
print("exported:", p)
mo = onnx.load(p)
print("inputs :", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in mo.graph.input])
print("outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in mo.graph.output])
