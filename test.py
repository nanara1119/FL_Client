import requests
from rest_framework.utils import json

data = [1, 2, 3,[4, 5, 6]]
data_to_json = json.dumps(data)

for i in range(5):
    result = requests.put("http://127.0.0.1:8000/weight/", data=data_to_json)
    print(result.text)