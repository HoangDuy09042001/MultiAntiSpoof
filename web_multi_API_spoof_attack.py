from flask import Flask, request, jsonify
import concurrent.futures
import time
from SpoofDetector1.web_API_spoof_attack import predict_spoof as predict_spoof1
from SpoofDetector2.web_API_spoof_attack import predict_spoof as predict_spoof2
from SpoofDetector3.web_API_spoof_attack import predict_spoof as predict_spoof3

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    st = time.time()
    PREDICT_SPOOF = [predict_spoof1, predict_spoof2, predict_spoof3]
    content = request.json
    image_urls = content['image_urls']
    if not image_urls:
        return jsonify({'error': 'No image URL provided'}), 400

    # Sử dụng ThreadPoolExecutor để thực hiện đa luồng
    with concurrent.futures.ThreadPoolExecutor() as executor:

        futures = [executor.submit(PREDICT_SPOOF[i], image_url) for i, image_url in enumerate(image_urls)]

        # Chờ cả hai công việc hoàn thành
        concurrent.futures.wait([future for future in futures])

        # Lấy kết quả từ hai công việc
        results = [future.result() for future in futures]
    result = {f'result{i}': result for i, result in enumerate(results)}
    print(time.time()-st)
    return jsonify({'predicted_label': result})

# def predict():
#     PREDICT_SPOOF = [predict_spoof1, predict_spoof2, predict_spoof3]
#     content = request.json
#     image_urls = content.get('image_urls', [])
#     if not image_urls:
#         return jsonify({'error': 'No image URLs provided'}), 400

#     results = {}

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         # Chạy predict_spoof cho mỗi image_url và tạo một future cho mỗi công việc
#         futures = {executor.submit(PREDICT_SPOOF[i], image_url, i): i for i, image_url in enumerate(image_urls)}

#         # Initialize i outside the loop
#         i = None

#         # Duyệt qua các future và lấy kết quả khi chúng hoàn thành
#         for future in concurrent.futures.as_completed(futures):
#             result = None
#             try:
#                 i = futures[future]
#                 result = future.result()
#             except Exception as e:
#                 results[f"result{i + 1}"] = f"Error: {e}"
#             if result is not None:
#                 results[f"result{i + 1}"] = result

#     return jsonify({'predicted_label': results})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4321, debug=True)