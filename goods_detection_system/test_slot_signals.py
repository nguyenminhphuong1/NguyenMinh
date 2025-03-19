import requests
import time
import argparse
import json
import os
import base64


def test_slot_signals(server_url="http://localhost:5000", force_init=False):
    """
    Kiểm thử API tín hiệu ô kệ đơn giản.

    Args:
        server_url: URL của server AI
        force_init: Buộc khởi tạo mô hình trước khi kiểm tra trạng thái
    """
    print(f"Kiểm thử API tín hiệu ô kệ tại {server_url}")

    # Kiểm tra kết nối tới server
    try:
        health_response = requests.get(f"{server_url}/health", timeout=5)
        if health_response.status_code != 200:
            print(f"Lỗi: Server không phản hồi đúng. Mã trạng thái: {health_response.status_code}")
            return

        print("Server đang hoạt động bình thường.")
    except requests.exceptions.RequestException as e:
        print(f"Lỗi kết nối đến server: {e}")
        return

    # Khởi tạo mô hình nếu cần
    if force_init:
        try:
            print("Đang khởi tạo mô hình...")

            # Sử dụng API endpoint /init_model mới để khởi tạo
            init_response = requests.post(f"{server_url}/api/init_model", timeout=10)

            if init_response.status_code == 200:
                print("Khởi tạo mô hình thành công.")
            else:
                print(f"Khởi tạo mô hình không thành công. Mã lỗi: {init_response.status_code}")
                print(f"Phản hồi: {init_response.text}")
                # Vẫn tiếp tục kiểm tra slots_status
        except Exception as e:
            print(f"Lỗi khi khởi tạo mô hình: {e}")
            # Vẫn tiếp tục kiểm tra slots_status

    # Gửi yêu cầu lấy trạng thái ô kệ
    try:
        response = requests.get(f"{server_url}/api/slots_status", timeout=5)

        if response.status_code == 200:
            data = response.json()
            signals = data.get("signals", [])

            print("\n--- TRẠNG THÁI Ô KỆ ---")
            print(f"Thời gian: {data.get('timestamp')}")
            print(f"Tổng số ô: {len(signals)}")

            if len(signals) == 0:
                print("Chưa có dữ liệu ô kệ. Cần chạy phát hiện trước.")
                if "note" in data:
                    print(f"Ghi chú: {data['note']}")
                suggest_detection(server_url)
                return

            filled_count = sum(1 for s in signals if s.get("has_goods", False))
            empty_count = len(signals) - filled_count

            print(f"Có hàng: {filled_count}")
            print(f"Trống: {empty_count}")

            # Hiển thị từng tín hiệu
            print("\nChi tiết tín hiệu:")
            for signal in signals:
                status = "CÓ HÀNG" if signal.get("has_goods", False) else "TRỐNG"
                print(f"Ô #{signal.get('id')}, Vị trí: {signal.get('position')}, Trạng thái: {status}")

            # Lưu kết quả ra file json để tham khảo
            with open("slot_signals_result.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                print("\nĐã lưu kết quả vào file slot_signals_result.json")

        elif response.status_code == 404:
            print("Chưa có dữ liệu phát hiện ô kệ. Hãy thực hiện phát hiện trước.")
            suggest_detection(server_url)
        elif response.status_code == 500:
            print(f"Lỗi máy chủ: {response.status_code}")
            print(f"Phản hồi: {response.text}")

            # Hỏi người dùng có muốn thử khởi tạo mô hình không
            if not force_init:
                try_init = input("Bạn có muốn thử khởi tạo mô hình không? (y/n): ")
                if try_init.lower() == 'y':
                    test_slot_signals(server_url, force_init=True)
                    return
        else:
            print(f"Lỗi: {response.status_code}")
            print(f"Phản hồi: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi lấy trạng thái ô kệ: {e}")


def suggest_detection(server_url):
    """Gợi ý người dùng chạy phát hiện ô kệ."""
    print("\nGợi ý: Cần chạy phát hiện ô kệ trước khi kiểm tra trạng thái.")
    print("Bạn có thể:")
    print("1. Chạy 'python test_slot_detection.py ./images/example.jpg' để cập nhật trạng thái")
    print("2. Hoặc gửi ảnh trực tiếp qua API /api/detect_slots")

    choice = input("\nBạn có muốn gửi ảnh mẫu để cập nhật trạng thái không? (y/n): ")

    if choice.lower() == 'y':
        # Tìm ảnh mẫu
        sample_dirs = ["./images", "../images", ".", ".."]
        sample_filenames = ["example.jpg", "sample.jpg", "shelf.jpg", "sample_shelf.jpg"]

        sample_path = None
        for directory in sample_dirs:
            for filename in sample_filenames:
                path = os.path.join(directory, filename)
                if os.path.isfile(path):
                    sample_path = path
                    break
            if sample_path:
                break

        if not sample_path:
            print("Không tìm thấy ảnh mẫu. Tạo ảnh tạm thời...")
            # Tạo ảnh đen đơn giản
            import numpy as np
            import cv2
            sample_path = "temp_sample.jpg"
            sample_image = np.zeros((800, 800, 3), dtype=np.uint8)
            cv2.imwrite(sample_path, sample_image)

        print(f"Sử dụng ảnh mẫu: {sample_path}")

        # Mã hóa và gửi ảnh
        try:
            with open(sample_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            print("Đang gửi ảnh để phát hiện ô kệ...")
            detect_response = requests.post(
                f"{server_url}/api/detect_slots",
                json={"image": encoded_image},
                timeout=30
            )

            if detect_response.status_code == 200:
                result = detect_response.json()
                print("\nPhát hiện thành công!")
                print(f"Tổng số ô kệ: {result['summary']['total_slots']}")
                print(f"Ô trống: {result['summary']['empty_slots']}")
                print(f"Ô có hàng: {result['summary']['filled_slots']}")
                print(f"Tỷ lệ lấp đầy: {result['summary']['occupancy_rate'] * 100:.1f}%")

                # Lưu ảnh kết quả
                if "result_image" in result:
                    result_image_data = base64.b64decode(result["result_image"])
                    result_image_path = "slot_detection_result.jpg"
                    with open(result_image_path, "wb") as f:
                        f.write(result_image_data)
                    print(f"Đã lưu ảnh kết quả vào {result_image_path}")

                # Thử lại kiểm tra trạng thái
                print("\nThử lại kiểm tra trạng thái ô kệ...")
                time.sleep(1)
                test_slot_signals(server_url, force_init=False)
            else:
                print(f"Lỗi khi phát hiện: {detect_response.status_code}")
                print(f"Phản hồi: {detect_response.text}")

        except Exception as e:
            print(f"Lỗi trong quá trình phát hiện: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm thử API tín hiệu ô kệ")
    parser.add_argument("--server", default="http://localhost:5000", help="URL của server AI")
    parser.add_argument("--init", action="store_true", help="Khởi tạo mô hình trước khi kiểm tra")

    args = parser.parse_args()
    test_slot_signals(args.server, args.init)