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
# تأكد أن ملف النموذج في GitHub باسم model.tflite
try:
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ تم تحميل نموذج TFLite بنجاح")
except Exception as e:
    print(f"❌ خطأ في تحميل النموذج: {e}")

# تحميل بيانات العلاج (تأكد من وجود الملف في GitHub)
with open('treatments.json', 'r', encoding='utf-8') as f:
    treatments_data = json.load(f)

# قائمة الفئات بالترتيب الذي تدرب عليه النموذج
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "online", "message": "Plant Doctor API is ready!"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image found"}), 400
    
    try:
        file = request.files['image']
        # معالجة الصورة لتناسب مدخلات النموذج (224x224)
        img = Image.open(file.stream).convert('RGB').resize((224, 224))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # تنفيذ التنبؤ
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        disease_key = CLASS_NAMES[class_idx]
        
        # جلب تفاصيل العلاج من ملف JSON
        result = treatments_data.get(disease_key, {
            "ar_name": "غير معروف",
            "medicine": "راجع مختصاً زراعياً",
            "usage": "-",
            "duration": "-"
        })
        
        result['confidence'] = f"{confidence * 100:.1f}%"
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # بورت ديناميكي ليتوافق مع Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
