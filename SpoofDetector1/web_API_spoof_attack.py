import os
from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import time

# Import the functions from your existing code
from SpoofDetector1.src.anti_spoof_predict import AntiSpoofPredict
from SpoofDetector1.src.generate_patches import CropImage
from SpoofDetector1.src.utility import parse_model_name
import requests
import base64
from io import BytesIO
from PIL import Image
from io import BytesIO


# Load the anti-spoofing models and initialize the AntiSpoofPredict object
MODEL_DIR = "/root/MultiAntiSpoofing/SpoofDetector1/resources/anti_spoof_models"
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

def predict_spoof(image_url):
    base64_data = image_url
    # Tách phần dữ liệu base64 từ chuỗi
    image_data = base64_data.split(';base64,')[-1]

    # Giải mã dữ liệu base64 và tạo đối tượng hình ảnh
    image_bytes = BytesIO(base64.b64decode(image_data))
    image = Image.open(image_bytes)

    # Lưu hình ảnh vào file
    output_file_path = 'SpoofDetector1/output_image.jpg'
    image.save(output_file_path, format='JPEG')

    # Đọc ảnh từ file
    image = cv2.imread(output_file_path)

    # Resize ảnh theo tỷ lệ 3/4 chiều dọc
    new_width = int(image.shape[0] * 3 / 4)
    # image = cv2.resize(image, (new_width, image.shape[0]))

    # Tính toán điểm cắt để cân đối từ giữa
    cut_width = (image.shape[1] - new_width) // 2

    # Thực hiện cắt để cân đối từ giữa
    image = image[:, cut_width:cut_width + new_width]
    output_cropped_file_path = 'SpoofDetector1/cropped_image.jpg'
    cv2.imwrite(output_cropped_file_path, image)
    # image.save(output_cropped_file_path, format='JPEG')

    image = cv2.imread(output_cropped_file_path)
    image = cv2.resize(image, (300, 400))
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

    # label = int(np.argmax(prediction))
    label = ((int(np.argmax(prediction))==1) or (abs(prediction[0][2] - prediction[0][1]) < 0.2)) 
    print('---------------------------------------------------------------------------')
    print('prediction[0][1] - prediction[0][2]: ', prediction[0][1], ' ',prediction[0][2], ' ', abs(prediction[0][1] - prediction[0][2]))
    print('label: ','REAL' if label==1 else 'FAKE',' ', prediction)
    result = False
    # face_recognize("Cruel_Summer/output_image.jpg")
    if(abs(prediction[0][1]-prediction[0][2]>0.2) or (label==1)):
        result = True
    return result
