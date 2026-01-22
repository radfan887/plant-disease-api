import os
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)
CORS(app)

# تحميل النموذج الخفيف TFLite
# تأكد أن اسم الملف في GitHub هو بالضبط model.tflite
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# قائمة الفئات بالترتيب الصحيح (يجب أن تطابق ترتيب التدريب)
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

# تحميل بيانات العلاج
with open('treatments.json', 'r', encoding='utf-8') as f:
    treatments_data = json.load(f)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "active", "message": "Plant Doctor API is running"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
            
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
        
        # جلب اسم الفئة
        disease_name = CLASS_NAMES[class_idx]

        # جلب بيانات العلاج باستخدام اسم الفئة
        result = treatments_data.get(disease_name, {
            "ar_name": "غير معروف",
            "medicine": "بيانات العلاج غير متوفرة لهذا التشخيص",
            "usage": "-",
            "duration": "-"
        })
        
        result['confidence'] = f"{confidence * 100:.1f}%"
        result['disease_key'] = disease_name
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # بورت 10000 هو الافتراضي لـ Render في كثير من الأحيان، لكن 5000 يعمل أيضاً
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
