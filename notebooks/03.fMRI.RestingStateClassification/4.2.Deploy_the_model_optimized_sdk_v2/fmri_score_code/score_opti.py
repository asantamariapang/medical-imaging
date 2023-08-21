from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import intel_extension_for_pytorch as ipex
import torchvision
from pathlib import Path
from data_split_25D import split_data
from fmri_utils import RGBDataset

import time
from openvino.runtime import Core
from openvino.runtime import get_version


def init():
    global target_device
    target_device = "cpu"

    # Initialize Test Data and Models.
    az_model_base_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "fmri-data-pt-onnx-ov-models"
    )

    data_dir = Path(os.path.join(az_model_base_path, "IC_niftis"))
    data_label_csv = Path(os.path.join(az_model_base_path, "ClusterNames.csv"))
    pt_model_path = os.path.join(az_model_base_path, "model_final25D.pth")

    model_pt = torch.load(pt_model_path, map_location=torch.device(target_device))
    subjects_test = model_pt["subjects_test"]
    label_names_unique = model_pt["label_names_unique"]
    print(len(subjects_test))

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
    global loader
    # load data into data loader for pytorchy stuff: #try to split into two steps
    loader = {
        x: DataLoader(dataset[x], num_workers=0, batch_size=batch_size, shuffle=True)
        for x in ["test"]
    }

    # Initiaize PyTorch model
    global modelx
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    #
    # Create model, loss function and optimzer
    #
    modelx = torchvision.models.resnet50(pretrained=True)
    modelx.fc = nn.Linear(2048, 58)  # Change the last layer to 58 output classes
    modelx.load_state_dict(model_pt["model"])
    modelx = modelx.to(target_device)
    modelx.eval()

    # Initial PyTorch IPEX model
    global ipex_modelx
    global traced_model
    ipex_modelx = ipex.optimize(modelx)

    # Initialize OpenVINO Runtime.
    global ov_compiled_model
    ov_core = Core()
    ov_xml = os.path.join(az_model_base_path, "fmri_model_final25D.xml")
    # Load and compile the OV model
    ov_model = ov_core.read_model(ov_xml)
    ov_compiled_model = ov_core.compile_model(model=ov_model, device_name=target_device.upper())


# TIP:  To accept raw data, use the AMLRequest class in your entry script and add the @rawhttp decorator to the run() function
#       more details in: https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-advanced-entry-script
# Note that despite the fact that we trained our model on PNGs, we would like to simulate
# a scenario closer to the real world here and accept DICOMs into our score script. Here's how:
@rawhttp
def run(request):
    if request.method == "GET":
        # For this example, just return the URL for GETs.
        respBody = str.encode(request.full_path)
        return AMLResponse(respBody, 200)

    elif request.method == "POST":
        # For a real-world solution, you would load the data from reqBody
        # and send it to the model. Then return the response.
        try:

            def benchmark_pt():
                phase = "test"
                test_accuracy = 0
                counter = 0
                correct = 0
                all_output = []
                all_predicted = []
                all_target = []

                print(
                    f"\n==== Benchmarking PyTorch inference with Test Data on CPU ===="
                )
                startTime = time.time()
                for index, subjects_batch in enumerate(loader[phase]):
                    data = subjects_batch[0]
                    data = data.to(target_device)
                    target = subjects_batch[1]
                    target = target.to(target_device)

                    output = modelx(data)

                    # Accuracy:
                    _, predicted = F.softmax(output, dim=1).max(1)
                    counter += data.size(0)
                    correct += predicted.eq(target).sum().item()
                    all_predicted += list(predicted.cpu().detach().numpy())
                    all_output += list(output.cpu().detach().numpy())
                    all_target += list(target.cpu().detach().numpy())

                print(f"PyTorch Summary:")
                # calculate average loss over an epoch
                print(f"Num subjects: {counter}, Loop index: {index}")
                test_accuracy = 100.0 * correct / counter
                print(f"Test Accuracy: {test_accuracy:1.0f}%")
                pt_time_sec = time.time() - startTime
                print(f"Time Taken : {pt_time_sec:.2f} seconds")

                # Return the result
                pt_summary = {
                    "fwk_version": f"PyTorch: {torch.__version__}",
                    "num_subjects": counter,
                    "test_accuracy": test_accuracy,
                    "time_sec": pt_time_sec,
                }
                return pt_summary

            def benchmark_ipex():
                phase = "test"
                test_accuracy = 0
                counter = 0
                correct = 0
                all_output = []
                all_predicted = []
                all_target = []

                for index, subjects_batch in enumerate(loader[phase]):
                    sample_batch = subjects_batch[0]
                    break

                with torch.no_grad():
                    traced_model = torch.jit.trace(ipex_modelx, sample_batch)
                    traced_model = torch.jit.freeze(traced_model)

                print(f"\n==== Benchmarking IPEX inference with Test Data on CPU ====")
                startTime = time.time()
                for index, subjects_batch in enumerate(loader[phase]):
                    data = subjects_batch[0]
                    data = data.to(target_device)
                    target = subjects_batch[1]
                    target = target.to(target_device)

                    with torch.no_grad():
                        output = traced_model(data)

                    # Accuracy:
                    _, predicted = F.softmax(output, dim=1).max(1)
                    counter += data.size(0)
                    correct += predicted.eq(target).sum().item()
                    all_predicted += list(predicted.cpu().detach().numpy())
                    all_output += list(output.cpu().detach().numpy())
                    all_target += list(target.cpu().detach().numpy())

                print(f"IPEX Summary:")
                # calculate average loss over an epoch
                print(f"Num subjects: {counter}, Loop index: {index}")
                test_accuracy = 100.0 * correct / counter
                print(f"Test Accuracy: {test_accuracy:1.0f}%")
                ipex_time_sec = time.time() - startTime
                print(f"Time Taken : {ipex_time_sec:.2f} seconds")

                # Return the result
                ipex_summary = {
                    "fwk_version": f"IPEX: {ipex.__version__}",
                    "num_subjects": counter,
                    "test_accuracy": test_accuracy,
                    "time_sec": ipex_time_sec,
                }
                return ipex_summary

            def benchmark_ov():
                phase = "test"
                test_accuracy = 0
                counter = 0
                correct = 0
                all_output = []
                all_predicted = []
                all_target = []

                input_layer = ov_compiled_model.input(0)
                output_layer = ov_compiled_model.output(0)

                print(
                    f"\n==== Benchmarking OpenVINO inference with Test Data on CPU ===="
                )
                startTime = time.time()
                for index, subjects_batch in enumerate(loader[phase]):
                    data = subjects_batch[0]
                    data = data.to(target_device)
                    target = subjects_batch[1]
                    target = target.to(target_device)

                    ov_output = ov_compiled_model(data)
                    ov_output = ov_output[output_layer]

                    # Accuracy:
                    _, ov_predicted = F.softmax(torch.from_numpy(ov_output), dim=1).max(1)
                    counter += data.size(0)
                    correct += ov_predicted.eq(target).sum().item()
                    all_predicted += list(ov_predicted.cpu().detach().numpy())
                    all_output += list(ov_output)
                    all_target += list(target.cpu().detach().numpy())

                print(f"OpenVINO Summary:")
                # calculate average loss over an epoch
                print(f"Num subjects: {counter}, Loop index: {index}")
                test_accuracy = 100.0 * correct / counter
                print(f"Test Accuracy: {test_accuracy:1.0f}%")
                ov_time_sec = time.time() - startTime
                print(f"Time Taken : {ov_time_sec:.2f} seconds")

                # Return the result
                ov_summary = {
                    "fwk_version": f"OpenVINO: {get_version()}",
                    "num_subjects": counter,
                    "test_accuracy": test_accuracy,
                    "time_sec": ov_time_sec,
                }
                return ov_summary

            # Get System information
            def get_system_info():
                import subprocess

                # Run lscpu command and capture output
                lscpu_out = subprocess.check_output(["lscpu"]).decode("utf-8")
                print(lscpu_out)
                # Run free -g command and capture output
                mem_out = subprocess.check_output(["free", "-g"]).decode("utf-8")
                print(mem_out)
                os_out = subprocess.check_output(["cat", "/etc/os-release"]).decode(
                    "utf-8"
                )
                kernal_out = subprocess.check_output(["uname", "-a"]).decode("utf-8")
                pyver_out = subprocess.check_output(["which", "python"]).decode("utf-8")
                os_out = os_out + " \n" + kernal_out + "\n" + pyver_out
                print(os_out)
                fwk_versions = {
                    "PyTorch": torch.__version__,
                    "IPEX": ipex.__version__,
                    "OpenVINO": get_version(),
                }
                print(fwk_versions)

                return_data = {
                    "lscpu_out": lscpu_out,
                    "mem_out_gb": mem_out,
                    "fwk_versions": fwk_versions,
                    "os": os_out,
                }
                return return_data

            #
            # Start Processing
            #
            # file_bytes = request.files["image"]

            # Benchmark PyTorch
            pt_summary = benchmark_pt()
            print(f"PyTorch Output: {pt_summary}")

            # Benchmark IPEX
            ipex_summary = benchmark_ipex()
            print(f"IPEX Output: {ipex_summary}")

            # Benchmark OpenVINO
            ov_summary = benchmark_ov()
            print(f"OpenVINO Output: {ov_summary}")

            sys_info = get_system_info()

            return_data = {
                "pt_summary": pt_summary,
                "ipex_summary": ipex_summary,
                "ov_summary": ov_summary,
                "system_info": sys_info,
            }

            return return_data

        except Exception as e:
            result = str(e)
            # return error message back to the client
            return AMLResponse(json.dumps({"error": result}), 200)

    else:
        return AMLResponse("bad request", 500)
