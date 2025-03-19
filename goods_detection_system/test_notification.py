import requests
import json
import time
import argparse


def test_notification(server_url="http://localhost:5000"):
    """
    Kiểm thử việc gửi thông báo trạng thái ô kệ thủ công.

    Args:
        server_url: URL của server AI
    """
    print(f"Kiểm thử gửi thông báo tới server tại: {server_url}")

    # Kiểm tra kết nối tới server
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

    # Giả lập dữ liệu trạng thái ô kệ
    slots_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera_id": "test_camera",
        "slots_status": [
            {
                "slot_id": 1,
                "has_goods": True,
                "position": {"row": 0, "column": 0},
                "confidence": 0.95
            },
            {
                "slot_id": 2,
                "has_goods": False,
                "position": {"row": 0, "column": 1},
                "confidence": 0.88
            },
            {
                "slot_id": 3,
                "has_goods": True,
                "position": {"row": 0, "column": 2},
                "confidence": 0.91
            },
            {
                "slot_id": 4,
                "has_goods": True,
                "position": {"row": 0, "column": 3},
                "confidence": 0.79
            },
            {
                "slot_id": 5,
                "has_goods": False,
                "position": {"row": 1, "column": 0},
                "confidence": 0.82
            },
            {
                "slot_id": 6,
                "has_goods": True,
                "position": {"row": 1, "column": 1},
                "confidence": 0.94
            }
        ]
    }

    # Tính toán tóm tắt
    filled_count = sum(1 for slot in slots_data["slots_status"] if slot["has_goods"])
    empty_count = len(slots_data["slots_status"]) - filled_count

    print(f"\nDữ liệu thử nghiệm:")
    print(f"Camera: {slots_data['camera_id']}")
    print(f"Thời gian: {slots_data['timestamp']}")
    print(f"Tổng số ô kệ: {len(slots_data['slots_status'])}")
    print(f"Số ô có hàng: {filled_count}")
    print(f"Số ô trống: {empty_count}")

    # Tạo yêu cầu thông báo
    notification_data = {
        "type": "slots_status",
        "data": slots_data
    }

    # Gửi yêu cầu
    print(f"\nĐang gửi thông báo trạng thái ô kệ...")
    try:
        response = requests.post(
            f"{server_url}/api/send_notification",
            json=notification_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        print(f"Mã trạng thái: {response.status_code}")
        if response.status_code == 200:
            print(f"Phản hồi: {response.json()}")
            print("\nThông báo đã được gửi thành công!")
        else:
            print(f"Lỗi: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi gửi thông báo: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kiểm thử gửi thông báo')
    parser.add_argument('--server', default='http://localhost:5000', help='URL của server AI')

    args = parser.parse_args()
    test_notification(args.server)