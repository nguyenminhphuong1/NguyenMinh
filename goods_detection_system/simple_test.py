import requests
import base64
import json
import sys

if len(sys.argv) < 2:
    print("Sử dụng: python simple_test.py <đường_dẫn_ảnh>")
    sys.exit(1)

# Đường dẫn đến tệp ảnh từ tham số dòng lệnh
image_path = sys.argv[1]

try:
    # Đọc file ảnh và mã hóa base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # Chuẩn bị dữ liệu cho yêu cầu
    data = {
        "image": encoded_string,
        "return_image": True
    }

    # Gửi yêu cầu POST đến endpoint /detect
    print(f"Đang gửi ảnh {image_path} tới server...")
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
            result_path = f"result_{image_path.split('/')[-1]}"
            with open(result_path, "wb") as f:
                f.write(base64.b64decode(result["result_image"]))
            print(f"Result image saved to '{result_path}'")
    else:
        print("Error:", response.status_code, response.text)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file ảnh tại '{image_path}'")
except Exception as e:
    print(f"Lỗi: {str(e)}")