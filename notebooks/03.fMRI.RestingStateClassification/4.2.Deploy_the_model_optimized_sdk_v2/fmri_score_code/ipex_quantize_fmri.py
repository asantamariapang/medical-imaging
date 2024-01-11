import torchvision
import torch.nn as nn
import torch
from pathlib import Path
from data_split_25D import split_data
from fmri_utils import RGBDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


# load model
target_device = "cpu"

pt_model_path = "model_final25D.pth"
model_pt = torch.load(pt_model_path, map_location=torch.device(target_device))

model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 58)  # Change the last layer to 58 output classes
model.load_state_dict(model_pt["model"])
model = model.to(target_device)
model.eval()

# set up val dataloader
data_dir = Path("IC_niftis")
data_label_csv = Path("ClusterNames.csv")

subjects_test = model_pt["subjects_test"]
label_names_unique = model_pt["label_names_unique"]

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

dataset = {"val": RGBDataset(subjects_valid, labels_val)}

batch_size = 1
loader = {
        x: DataLoader(dataset[x], num_workers=0, batch_size=batch_size, shuffle=True)
        for x in ["val"]
    }
val_loader = loader["val"]

for index,subjects_batch  in enumerate(val_loader):
    data = subjects_batch[0]
    target = subjects_batch[1]
    break

import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert

qconfig = ipex.quantization.default_static_qconfig
prepared_model = prepare(model, qconfig, example_inputs=data, inplace=False)

# run calibration steps for static quantization
for d in val_loader:
  prepared_model(d[0])

converted_model = convert(prepared_model)

# convert to a torchscript module
with torch.no_grad():
  traced_model = torch.jit.trace(converted_model, data)
  traced_model = torch.jit.freeze(traced_model)

# save traced model
traced_model.save("quantized_model.pt")
