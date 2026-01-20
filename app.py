from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing import image
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app)  # للسماح لتطبيق Flutter بالوصول إلى الخادم

# 1. إعداد المسارات (تأكد من وجود هذه الملفات في نفس المجلد)
MODEL_PATH = 'plant_model.h5'
JSON_PATH = 'treatments.json'

# 2. قائمة الفئات (يجب أن تطابق ترتيب التدريب الخاص بك)
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
    'Tomato_healthy'
]

# 3. تحميل النموذج وبيانات العلاج عند بدء السيرفر
print("⏳ جاري تحميل النموذج والبيانات...")
model = tf.keras.models.load_model(MODEL_PATH)

if os.path.exists(JSON_PATH):
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        treatments_db = json.load(f)
else:
    treatments_db = {}
    print("⚠️ تحذير: ملف treatments.json غير موجود!")

@app.route('/', methods=['GET'])
def index():
    return "سيرفر طبيب النبات الذكي يعمل بنجاح!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # التأكد من استقبال صورة
        if 'image' not in request.files:
            return jsonify({'error': 'لم يتم استلام أي صورة'}), 400
        
        file = request.files['image']
        
        # معالجة الصورة لتحويلها إلى مصفوفة يفهمها النموذج
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # إجراء التنبؤ
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions)
        confidence = float(np.max(predictions))
        
        predicted_class = CLASS_NAMES[class_idx]

        # جلب معلومات العلاج من الـ JSON
        treatment_info = treatments_db.get(predicted_class, {
            "ar_name": "غير متوفر",
            "medicine": "لا يوجد بيانات علاجية حالياً",
            "usage": "-",
            "duration": "-",
            "expectations": "-"
        })

        # إرسال النتيجة النهائية للتطبيق
        return jsonify({
            'status': 'success',
            'disease_key': predicted_class,
            'confidence': f"{confidence * 100:.2f}%",
            'ar_name': treatment_info['ar_name'],
            'medicine': treatment_info['medicine'],
            'usage': treatment_info['usage'],
            'duration': treatment_info['duration'],
            'expectations': treatment_info.get('expectations', '-')
        })

    except Exception as e:
        print(f"❌ خطأ: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # تشغيل السيرفر على الشبكة المحلية لكي يراه الهاتف
    app.run(host='0.0.0.0', port=5000, debug=True)