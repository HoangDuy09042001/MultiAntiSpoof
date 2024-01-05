import os
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import time

# Import the functions from your existing code
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import requests
import base64
from io import BytesIO
from PIL import Image
from io import BytesIO
# from reputation import face_recognize
app = Flask(__name__)

# Load the anti-spoofing models and initialize the AntiSpoofPredict object
MODEL_DIR = "/home/hungha/AI_365/Cruel_Summer/resources/anti_spoof_models"
DEVICE_ID = 0
model_test = AntiSpoofPredict(DEVICE_ID)

# Initialize the image_cropper object for cropping the image
image_cropper = CropImage()

# Define a function to check the image dimensions
def check_image(image):
    height, width, channel = image.shape
    if width / height != 3 / 4:
        return False
    return True


@app.route('/predict', methods=['POST'])
def predict():
    image_url = request.form.get('image_url')
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400
    base64_data = image_url
    # Tách phần dữ liệu base64 từ chuỗi
    image_data = base64_data.split(';base64,')[-1]

    # Giải mã dữ liệu base64 và tạo đối tượng hình ảnh
    image_bytes = BytesIO(base64.b64decode(image_data))
    image = Image.open(image_bytes)

    # Lưu hình ảnh vào file
    output_file_path = '/home/hungha/AI_365/Cruel_Summer/output_image.jpg'
    image.save(output_file_path, format='JPEG')

    # print(f'Hình ảnh đã được lưu tại {output_file_path}')



    # Đường dẫn đến ảnh đã lưu
    saved_image_path = '/home/hungha/AI_365/Cruel_Summer/output_image.jpg'

    # Đọc ảnh từ file
    image = cv2.imread(saved_image_path)

    # image = cv2.resize(image, (int(image.shape[0] * 3 / 4), image.shape[0]))
    image = cv2.resize(image, (int(image.shape[0] * 3), int(image.shape[0] * 4)))
    result = check_image(image)
    if not result:
        return jsonify({'error': 'Image is not appropriate. Height/Width should be 4/3.'}), 400

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0

    # Sum the prediction from single model's result
    for model_name in os.listdir(MODEL_DIR):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(MODEL_DIR, model_name))
        test_speed += time.time() - start

    # Draw result of prediction
    # print('prediction: ',prediction)
    label = int(np.argmax(prediction))
    print('---------------------------------------------------------------------------')
    print('prediction[0][1] - prediction[0][2]: ', prediction[0][1], ' ',prediction[0][2], ' ', abs(prediction[0][1] - prediction[0][2]))
    print('label: ','REAL' if label==1 else 'FAKE',' ', prediction)
    result=False
    # face_recognize("Cruel_Summer/output_image.jpg")
    if(abs(prediction[0][1]-prediction[0][2]>0.2) or (label==1)):
        result=True
    value = prediction[0][label] / 2
    return jsonify({'predicted_label': result})

    # return 'Hello'


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4321)

