import requests

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

# enter details of your AML workspace
subscription_id = "1e9b4bc4-253a-40c3-8771-998507855894"
resource_group = "user-manageg-key-test"
workspace = "nuance_benchmark"

# get a handle to the workspace
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)

endpoint_name = "fmri-pt-ipex-ov-sdk-v2-1"
deployed_endpoint = ml_client.online_endpoints.get(name=endpoint_name)

# existing traffic details
print(deployed_endpoint.traffic)
# Get the scoring URI
scoring_uri = deployed_endpoint.scoring_uri
print(scoring_uri)
auth_key = ml_client.online_endpoints.get_keys(endpoint_name).primary_key
print(f"Authkye:{auth_key}")

# resp = requests.post(scoring_uri, input_data, headers=headers)

print(f"Sending request to {scoring_uri}")
# Send the HTTP request and obtain results from endpoint.
response = requests.post(scoring_uri, headers={"Authorization": f"Bearer {auth_key}"}, timeout=600)
print("output:", response.content)
