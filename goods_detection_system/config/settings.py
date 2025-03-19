import os
import json
import platform

# Đường dẫn cơ sở đến thư mục dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Kiểm tra hệ điều hành
IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# Cấu hình AI Server
AI_SERVER = {
    "host": "localhost",
    "port": 5000,
    "model_path": os.path.join(BASE_DIR, "models", "weights", "fast_rcnn_model.pth"),
    "threshold": 0.5,  # Ngưỡng phát hiện
    "device": "cpu"  # hoặc "cuda" nếu có GPU
}

# Cấu hình thông báo server
NOTIFICATION_SERVER = {
    "url": "http://localhost:9000/notification",  # URL của server nhận thông báo
    "api_key": "your_api_key_here",  # API key để xác thực với server nhận thông báo
    "retry_attempts": 3,  # Số lần thử lại nếu gửi thất bại
    "timeout": 5,  # Thời gian chờ (giây)

    # Cấu hình thêm cho thông báo ô kệ
    "slots_notification": {
        "enabled": True,  # Bật/tắt tính năng thông báo ô kệ
        "min_interval": 1.0,  # Khoảng thời gian tối thiểu giữa các thông báo (giây)
        "send_on_change_only": True,  # Chỉ gửi thông báo khi có sự thay đổi
        "include_image": False  # Có bao gồm hình ảnh trong thông báo hay không
    }
}

# Cấu hình giao diện người dùng
UI_CONFIG = {
    "window_title": "Hệ thống phát hiện hàng hóa",
    "window_width": 1280,
    "window_height": 720,
    "update_interval": 100,  # Cập nhật UI mỗi 100ms
    "max_fps": 30,
    "debug_mode": os.environ.get("DEBUG", "0") == "1"
}

# Cấu hình ghi log
LOG_CONFIG = {
    "log_dir": os.path.join(BASE_DIR, "logs"),
    "max_log_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "log_level": "DEBUG" if os.environ.get("DEBUG", "0") == "1" else "INFO"
}


def load_camera_config():
    """Tải cấu hình camera từ tệp JSON."""
    config_path = os.path.join(BASE_DIR, "config", "camera_config.json")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Trả về cấu hình mặc định nếu không tìm thấy tệp
        return {
            "cameras": [
                {
                    "id": "main_camera",
                    "name": "Camera chính",
                    "type": "webcam",
                    "device_id": 0,
                    "enabled": True
                }
            ]
        }


# Tải cấu hình camera
CAMERA_CONFIG = load_camera_config()