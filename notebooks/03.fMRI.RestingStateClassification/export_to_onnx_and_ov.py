
import torch
import torch.nn as nn
import torchvision
import openvino as ov
from openvino.tools.mo import convert_model

model_final =  torch.load('outputs/model_final25D.pth', map_location=torch.device('cpu'))

model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 58) # Change the last layer to 58 output classes

device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f'Device is {device}')

model.load_state_dict(model_final['model'])
model = model.to(device)
model.eval()

# export to ONNX
dummy_input = torch.randn(8, 3, 45, 45)
onnx_fn = "fmri_model_final25D.onnx"
print(f"Exporting to {onnx_fn}...")
torch.onnx.export(model, dummy_input, onnx_fn, opset_version=16)
print(f"Done exporting to {onnx_fn}")

# export to OPENVINO IR
ov_fn = "fmri_25D_ir/fmri_model_final25D.xml"
print(f"Exporting to {ov_fn}...")
ov_model = convert_model(onnx_fn, input_shape=(8, 3, 45, 45))
ov.runtime.serialize(ov_model, xml_path=ov_fn)
print(f"Done exporting to {ov_fn}")