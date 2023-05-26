import json
import requests
headers = {"Authorization": f"Bearer "}
API_URL = "https://api-inference.huggingface.co/models/hustvl/yolos-tiny"
def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.request("POST", API_URL, headers=headers, data=data)
    print(response)
    return json.loads(response.content.decode("utf-8"))
data = query("examples/example_1.jpg")

print(data)