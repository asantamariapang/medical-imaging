import os
import torch
import torch.nn as nn
import torchvision
import openvino as ov
from openvino.tools.mo import convert_model

base_model_name =  "model_final25D"

# az_model_base_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "fmri-data-pt-onnx-ov-models")
az_model_base_path = os.path.join("fmri-data-pt-onnx-ov-models")

pt_model_path = os.path.join(az_model_base_path, f"{base_model_name}.pth")
onnx_model_path = os.path.join(az_model_base_path, f"{base_model_name}.onnx")
ov_model_path = os.path.join(az_model_base_path, f"{base_model_name}.xml")

# Load PyTorch Model
print(f"Loading PyTorch Model: {pt_model_path}")
model_final =  torch.load(pt_model_path, map_location=torch.device('cpu'))

model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 58) # Change the last layer to 58 output classes

device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f'Device is {device}')

model.load_state_dict(model_final['model'])
model = model.to(device)
model.eval()
print(f"PyTorch Model loaded and set to EVAL.")

# Export to ONNX
dummy_input = torch.randn(8, 3, 45, 45)

print(f"Exporting to ONNX: {onnx_model_path}")
torch.onnx.export(model, dummy_input, onnx_model_path, opset_version=16)
print(f"Done exporting to {onnx_model_path}")

# Export to OPENVINO IR
print(f"Exporting to OpenVINO IR: {ov_model_path}")
ov_model = convert_model(onnx_model_path, input_shape=(8, 3, 45, 45))
ov.runtime.serialize(ov_model, xml_path=ov_model_path)
print(f"Done exporting to {ov_model_path}")