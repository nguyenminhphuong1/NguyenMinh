import requests
import base64
import json

# Đường dẫn đến tệp ảnh - thay đổi thành ảnh của bạn
# image_path = "goods_detection_system/images/20250120_20250120140211_20250120143422_140210_1_1.jpg"
image_path = sys.argv[1] if len(sys.argv) > 1 else "images/test_image.jpg"
# Đọc file ảnh và mã hóa base64
with open(image_path, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Chuẩn bị dữ liệu cho yêu cầu
data = {
    "image": encoded_string,
    "return_image": True
}

# Gửi yêu cầu POST đến endpoint /detect
response = requests.post("http://localhost:5000/detect", json=data)

# Kiểm tra và in kết quả
if response.status_code == 200:
    result = response.json()
    print("Status:", "Success" if result.get("success") else "Failed")
    print("Has goods:", result.get("has_goods"))
    print("Confidence:", result.get("confidence"))

    # In thông tin về các phát hiện
    detections = result.get("detections", [])
    print(f"Found {len(detections)} object(s):")
    for i, detection in enumerate(detections):
        print(f"  {i + 1}. Class: {detection['class']}, Confidence: {detection['confidence']:.2f}")

    # Lưu ảnh kết quả nếu có
    if "result_image" in result:
        with open("result.jpg", "wb") as f:
            f.write(base64.b64decode(result["result_image"]))
        print("Result image saved to 'result.jpg'")
else:
    print("Error:", response.status_code, response.text)