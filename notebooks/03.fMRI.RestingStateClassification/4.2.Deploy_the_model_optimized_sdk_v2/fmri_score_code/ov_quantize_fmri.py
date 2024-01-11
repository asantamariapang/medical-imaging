import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from data_split_25D import split_data
from fmri_utils import RGBDataset

from openvino.runtime import Core, serialize
from openvino.runtime import get_version
from openvino.tools import mo
import nncf

device = 'cuda' if torch.cuda.is_available() else "cpu"
print(f'Device is {device}')

# az_model_base_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "fmri-data-pt-onnx-ov-models")
az_model_base_path = os.path.join("..","..", "fmri-data-pt-onnx-ov-models")

base_model_name =  "model_final25D"
data_dir = Path(os.path.join(az_model_base_path, "IC_niftis"))
data_label_csv = Path(os.path.join(az_model_base_path, "ClusterNames.csv"))

pt_model_path = os.path.join(az_model_base_path, f"{base_model_name}.pth")
onnx_model_path = os.path.join(az_model_base_path, f"{base_model_name}.onnx")
ov_model_path = os.path.join(az_model_base_path, f"{base_model_name}.xml")

int8_onnx_path = az_model_base_path / Path(base_model_name + "_int8").with_suffix(".onnx")
int8_ir_path = az_model_base_path / Path(base_model_name + "_int8").with_suffix(".xml")
#
# Load PT Model
#
print(f"Loading PyTorch Model: {pt_model_path}")
model_pt =  torch.load(pt_model_path, map_location=torch.device('cpu'))

#
# Setup DataLoader
#
print(f"Setting up DataLoader: {data_dir} ")
subjects_test = model_pt["subjects_test"]
label_names_unique = model_pt["label_names_unique"]
print(f"Length of subjects_test: {len(subjects_test)}")

(
    subjects_train,
    subjects_valid,
    subjects_test,
    labels_train,
    labels_val,
    labels_test,
    flat_labels,
    class_weights,
    label_names_unique,
) = split_data(data_dir, data_label_csv)

dataset = {"test": RGBDataset(subjects_test, labels_test)}
print(f"Test Dataset Length: {len(dataset['test'])}")

batch_size = 8
# load data into data loader for pytorchy stuff: #try to split into two steps
loader = {
    x: DataLoader(dataset[x], num_workers=0, batch_size=batch_size, shuffle=True)
    for x in ["test"]
}
print(f"DataLoader initialized... ")

#
# Load load_state_dict into PT model
#
print(f"Loading load_state_dict into PyTorch Model... ")
model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 58) # Change the last layer to 58 output classes

model.load_state_dict(model_pt['model'])
model = model.to(device)
model.eval()

print(f"PyTorch model loaded and set to eval()")

def transform_fn(data_item):
    images = data_item[0]
    return images

# Setup Calibration Dataset
calibration_dataset = nncf.Dataset(loader["test"], transform_fn)
print(f"NNCF calibration_dataset initialized")

# Quantize the model
quantized_model = nncf.quantize(model, calibration_dataset)
print(f"NNCF Quantization complete")

# Save INT8 ONNX and IR model
dummy_input = torch.randn(8, 3, 45, 45)
torch.onnx.export(quantized_model,
                  dummy_input,
                  int8_onnx_path,)

quantized_model_ir = mo.convert_model(input_model=int8_onnx_path)
serialize(quantized_model_ir, str(int8_ir_path))

print(f"OpenVINO version: {get_version()}")
print(f"ONNX Quantized models saved at {int8_onnx_path} ")
print(f"OpenVINO Quantized models saved at {int8_ir_path} ")
