import urllib.request, json, os

url = 'http://127.0.0.1:8000/predict'
img_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_images', 'fantasy_tiger.jpg')

boundary = b'----FormBoundary7MA4YWxkTrZu0gW'
with open(img_path, 'rb') as f:
    img_data = f.read()

body = (
    b'------FormBoundary7MA4YWxkTrZu0gW\r\n'
    b'Content-Disposition: form-data; name="file"; filename="fantasy_tiger.jpg"\r\n'
    b'Content-Type: image/jpeg\r\n\r\n'
    + img_data +
    b'\r\n------FormBoundary7MA4YWxkTrZu0gW--\r\n'
)

req = urllib.request.Request(url, data=body)
req.add_header('Content-Type', 'multipart/form-data; boundary=----FormBoundary7MA4YWxkTrZu0gW')
req.add_header('Content-Length', str(len(body)))

try:
    with urllib.request.urlopen(req, timeout=90) as resp:
        result = json.loads(resp.read())
        print('STATUS:', result.get('status'))
        pr = result.get('primary_result', {})
        print('PREDICTION:', pr.get('prediction'))
        print('CONFIDENCE:', pr.get('confidence'))
        print('IMAGE TYPE:', pr.get('image_type'))
        print('MODEL USED:', pr.get('model_used'))
        print()
        print('Full primary result:')
        print(json.dumps(pr, indent=2))
except Exception as e:
    print('ERROR:', e)
