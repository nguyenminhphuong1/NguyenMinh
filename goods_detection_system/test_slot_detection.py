import requests
import base64
import json
import sys
import os
import time
import argparse


def test_slot_detection(image_path, server_url="http://localhost:5000"):
    """
    Kiểm tra chức năng phát hiện ô kệ trong kho hàng.

    Args:
        image_path: Đường dẫn đến tệp ảnh
        server_url: URL của server AI
    """
    print(f"Kiểm tra phát hiện ô kệ với ảnh: {image_path}")

    # Kiểm tra xem file ảnh có tồn tại không
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy file ảnh tại {image_path}")
        return

    # Kiểm tra xem server có hoạt động không
    try:
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"Lỗi: Server không phản hồi đúng. Mã trạng thái: {health_response.status_code}")
            print(f"Phản hồi: {health_response.text}")
            return

        print("Server đang hoạt động bình thường.")
        print(f"Trạng thái: {health_response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Lỗi kết nối đến server: {e}")
        return

    # Đọc file ảnh và mã hóa base64
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Lỗi khi đọc file ảnh: {e}")
        return

    # Chuẩn bị dữ liệu cho yêu cầu
    data = {
        "image": encoded_string,
        "detect_slots": True,  # Bật chế độ phát hiện ô kệ
        "return_image": True
    }

    # Gửi yêu cầu POST đến endpoint /detect
    print("\nĐang gửi yêu cầu phát hiện ô kệ...")
    try:
        start_time = time.time()
        response = requests.post(f"{server_url}/detect", json=data, timeout=30)
        request_time = time.time() - start_time

        print(f"Thời gian yêu cầu: {request_time:.2f} giây")
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi gửi yêu cầu: {e}")
        return

    # Kiểm tra và in kết quả
    if response.status_code == 200:
        try:
            result = response.json()

            print("\n--- KẾT QUẢ PHÁT HIỆN Ô KỆ ---")
            print(f"Trạng thái: {'Thành công' if result.get('success') else 'Thất bại'}")

            # Hiển thị thông tin tổng kết
            if "summary" in result:
                summary = result["summary"]
                print(f"\nTỔNG KẾT:")
                print(f"  Tổng số ô kệ: {summary.get('total_slots', 0)}")
                print(f"  Số ô trống: {summary.get('empty_slots', 0)}")
                print(f"  Số ô có hàng: {summary.get('filled_slots', 0)}")
                print(f"  Tỷ lệ lấp đầy: {summary.get('occupancy_rate', 0):.2%}")

            # In thông tin về các phát hiện
            detections = result.get("detections", [])
            print(f"\nĐã phát hiện {len(detections)} ô kệ:")

            # Phân loại theo dãy
            slots_by_row = {}
            for detection in detections:
                row = detection.get("row", 0)
                if row not in slots_by_row:
                    slots_by_row[row] = []
                slots_by_row[row].append(detection)

            # In theo dãy
            for row in sorted(slots_by_row.keys()):
                print(f"\nDãy {row}:")
                slots = slots_by_row[row]
                # Sắp xếp theo cột
                slots.sort(key=lambda x: x.get("column", 0))

                for slot in slots:
                    slot_id = slot.get("slot_id", 0)
                    class_name = slot.get("class", "unknown")
                    status = "Có hàng" if class_name == "filled_slot" else "Trống"
                    confidence = slot.get("confidence", 0)
                    column = slot.get("column", 0)

                    print(f"  Ô #{slot_id} (cột {column}): {status} (tin cậy: {confidence:.2%})")

            # Lưu ảnh kết quả nếu có
            if "result_image" in result:
                output_path = os.path.splitext(image_path)[0] + "_slot_result.jpg"
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(result["result_image"]))
                print(f"\nHình ảnh kết quả đã được lưu tại: {output_path}")

        except json.JSONDecodeError:
            print("Lỗi: Không thể phân tích phản hồi JSON")
            print(f"Nội dung phản hồi: {response.text[:200]}...")
    else:
        print(f"Lỗi: Mã trạng thái HTTP {response.status_code}")
        print(f"Phản hồi: {response.text}")


if __name__ == "__main__":
    # Tạo parser dòng lệnh
    parser = argparse.ArgumentParser(description='Kiểm tra chức năng phát hiện ô kệ')
    parser.add_argument('image_path', help='Đường dẫn đến tệp ảnh để phát hiện')
    parser.add_argument('--server', default='http://localhost:5000',
                        help='URL của server AI (mặc định: http://localhost:5000)')

    args = parser.parse_args()

    # Chạy kiểm tra
    test_slot_detection(args.image_path, args.server)