from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
# สร้างแอป Flask
app = Flask(__name__)

knn = joblib.load('knn_model.pkl')
svm = joblib.load('svm_model.pkl')
# สร้าง route สำหรับ API
CORS(app,origins="*")
@app.route('/')
def hello_world():
    return jsonify(message=[0,1,2,3])

# กำหนด route สำหรับทำนาย (predict)
@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    input_data = None
    try:
        data = request.get_json()
        
        if data is None:
            return jsonify({'error': 'Invalid JSON format or no data provided'}), 400
        
        if 'input' not in data:
            return jsonify({'error': 'No input data provided'}), 400
        
        input_data = np.array(data['input']).reshape(1, -1)
        prediction = knn.predict(input_data)
        print(f"Input data: {input_data}")
        print(f"Prediction: {prediction}")
        print(prediction)
        return jsonify(prediction=prediction.tolist())

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    try:
        data = request.get_json()

        if 'input' not in data:
            return jsonify({'error': 'No input data provided'}), 400
        
        input_data = np.array(data['input']).reshape(1, -1)
        
        prediction = svm.predict(input_data)

        return jsonify(prediction=prediction.tolist())

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)