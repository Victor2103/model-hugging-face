import requests
from gradio import processing_utils

with open("examples/example_1.jpg", "rb") as image_file:
    encoded_string = processing_utils.encode_url_or_file_to_base64(image_file)

headers={'Accept': 'application/json'}

body={
    "fn_index": 0,
    "data": [
        f"data:image/jpeg;base64,{encoded_string}"
    ],
    "event_data": None,
    "session_hash": "2zx2slhv3qm"
}

res=requests.post("https://7f3b114c-120b-42ac-88e4-84982e9e4bc0.app.gra.ai.cloud.ovh.net/run/predict",json=body,headers=headers)

print(body)
print(res.text)










"""
headers={'Accept': 'application/json',
         'Authorization': f'Bearer SAnNu2i6R+K6JOYaAUDclnTWx1XH3Ck7+hIfr2dWj4oLEbt+XvhMnvUUegm0QFY9'}

body={"fn_index":0,"data":["can you"],"event_data":None,"session_hash":"q75l89if3v8"}

res=requests.post("https://f29f35cd-dce1-4a16-8cbc-c1f548cc7a46.app.gra.ai.cloud.ovh.net/run/predict",json=body,headers=headers)

print(res.text)
"""