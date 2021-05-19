from flask import jsonify
from google.cloud import storage
import joblib

storage_client = storage.Client()
bucket = storage_client.get_bucket('my_testing_tmp_files')
blob = bucket.blob('sentiment_model.save')
blob.download_to_filename('/tmp/sentiment_model.save')
model = joblib.load('/tmp/sentiment_model.save')

label_to_text = {
  -1: 'Negative',
  0: 'Neutral',
  1: 'Positive',
  2: 'Not found'}

def predict(request):
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }

        return ('', 204, headers)
    
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    request_json = request.get_json(force=True, silent=True)
    
    if request.args and 'text' in request.args:
        pred = model.predict([request.args.get('text')])
    elif request_json and 'text' in request_json:
        pred = model.predict([request_json['text']])
    else:
        pred = [2]
    
    resp = jsonify({'sentiment': label_to_text[pred[0]]})
    resp.headers['Access-Control-Allow-Origin'] = '*'

    return resp
