import os
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# تحميل النموذج الخفيف TFLite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# تحميل بيانات العلاج
with open('treatments.json', 'r', encoding='utf-8') as f:
    treatments = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB').resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # التنبؤ باستخدام TFLite
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    result = treatments.get(str(class_idx), treatments["0"])
    result['confidence'] = f"{confidence * 100:.1f}%"
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)