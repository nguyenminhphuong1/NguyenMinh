import requests
import base64
import sys


def init_slots(image_path):
    """Gửi ảnh để phát hiện ô kệ và khởi tạo dữ liệu trên server."""
    # Đọc ảnh
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Lỗi khi đọc file ảnh: {e}")
        return False

    # Gửi yêu cầu phát hiện ô kệ
    try:
        response = requests.post(
            "http://localhost:5000/detect",
            json={
                "image": encoded_string,
                "detect_slots": True,
                "camera_id": "test_camera"
            },
            timeout=30  # Tăng timeout vì phát hiện có thể mất thời gian
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Đã phát hiện {len(result.get('detections', []))} ô kệ")
            print(f"Trạng thái: {'Thành công' if result.get('success') else 'Thất bại'}")
            return True
        else:
            print(f"Lỗi: HTTP {response.status_code}")
            print(f"Phản hồi: {response.text}")
            return False
    except Exception as e:
        print(f"Lỗi khi gửi yêu cầu: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Sử dụng: python init_slots.py ./images/example.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    success = init_slots(image_path)

    if success:
        print("Đã khởi tạo dữ liệu ô kệ thành công. Bây giờ bạn có thể chạy test_slot_signals.py")
    else:
        print("Không thể khởi tạo dữ liệu ô kệ")