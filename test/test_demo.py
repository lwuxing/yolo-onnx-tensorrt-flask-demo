import requests
import base64
import time

HEADERS = {"Content-Type": "application/json"}
url = "http://192.168.175.4:10086/v1/object-detection"
image_path = "fall_0.jpg"


with open(image_path, 'rb') as f:
    img_data = f.read()
    base64_data = base64.b64encode(img_data)
    base64_data = base64_data.decode()

data = {
    "uid": "123456789",
    "data": base64_data,
    "fmt": 'jpg',
    "param": "{}"
}

tic = time.time()
for _ in range(100):
    tic = time.time()
    response = requests.post(url, headers=HEADERS, json=data)
    print("request:", response.text)
print('time: ', (time.time() - tic) / 100)
# tensorrt time:  0.0003009319305419922
# onnx time:  0.0008700418472290039