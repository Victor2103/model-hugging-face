import requests

headers={'Accept': 'application/json',
         'Authorization': f'Bearer SAnNu2i6R+K6JOYaAUDclnTWx1XH3Ck7+hIfr2dWj4oLEbt+XvhMnvUUegm0QFY9'}

body={"fn_index":0,"data":["can you"],"event_data":None,"session_hash":"q75l89if3v8"}

res=requests.post("https://f29f35cd-dce1-4a16-8cbc-c1f548cc7a46.app.gra.ai.cloud.ovh.net/run/predict",json=body,headers=headers)

print(res.text)