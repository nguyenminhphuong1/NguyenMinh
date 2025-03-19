import cv2
import requests
import base64
import time
import logging
import json
import os
from datetime import datetime
import numpy as np
import threading

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("camera_ip.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IPCamera:
    def __init__(self, camera_id, rtsp_url, username=None, password=None, ai_server_url="http://localhost:5000"):
        """
        Khởi tạo kết nối camera IP.

        Args:
            camera_id (str): ID duy nhất của camera
            rtsp_url (str): URL RTSP của camera IP (ví dụ: rtsp://192.168.1.100:554/stream)
            username (str, optional): Tên đăng nhập nếu camera yêu cầu xác thực
            password (str, optional): Mật khẩu nếu camera yêu cầu xác thực
            ai_server_url (str): URL của AI server để gửi hình ảnh đến
        """
        self.camera_id = camera_id
        self.ai_server_url = ai_server_url

        # Xử lý xác thực trong URL nếu cần
        if username and password:
            # Chèn thông tin đăng nhập vào URL RTSP
            protocol = rtsp_url.split("://")[0]
            address = rtsp_url.split("://")[1]
            self.rtsp_url = f"{protocol}://{username}:{password}@{address}"
        else:
            self.rtsp_url = rtsp_url

        self.cap = None
        self.is_connected = False
        self.is_running = False
        self.last_frame = None
        self.last_frame_time = None
        self.frame_count = 0
        self.detection_thread = None

    def connect(self):
        """Kết nối với camera IP qua RTSP."""
        try:
            logger.info(f"Đang kết nối đến camera IP: {self.camera_id} - {self.rtsp_url}")

            # Một số tùy chọn để cải thiện hiệu suất với RTSP
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

            # Thiết lập kết nối với camera
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            # Kiểm tra kết nối
            if not self.cap.isOpened():
                logger.error(f"Không thể kết nối đến camera: {self.rtsp_url}")
                return False

            # Đọc một frame để xác nhận kết nối hoạt động
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Không thể đọc frame từ camera, kiểm tra lại kết nối")
                self.cap.release()
                return False

            # Lưu frame đầu tiên và thời gian
            self.last_frame = frame
            self.last_frame_time = datetime.now()
            self.is_connected = True
            logger.info(f"Kết nối thành công đến camera IP: {self.camera_id}")

            return True

        except Exception as e:
            logger.error(f"Lỗi khi kết nối đến camera IP: {e}")
            if self.cap and self.cap.isOpened():
                self.cap.release()
            return False

    def disconnect(self):
        """Ngắt kết nối camera."""
        self.is_running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)

        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.is_connected = False
        logger.info(f"Đã ngắt kết nối camera: {self.camera_id}")

    def start_capture(self, interval=1.0, detect_slots=True):
        """
        Bắt đầu quá trình ghi hình liên tục và gửi đến AI server.

        Args:
            interval (float): Khoảng thời gian giữa các lần phát hiện (giây)
            detect_slots (bool): Có phát hiện trạng thái ô kệ hay không
        """
        if not self.is_connected:
            success = self.connect()
            if not success:
                return False

        if self.is_running:
            logger.warning("Quá trình ghi hình đã đang chạy")
            return True

        self.is_running = True
        self.detection_thread = threading.Thread(
            target=self._capture_and_detect_loop,
            args=(interval, detect_slots),
            daemon=True
        )
        self.detection_thread.start()
        logger.info(f"Đã bắt đầu ghi hình từ camera {self.camera_id}")
        return True

    def _capture_and_detect_loop(self, interval, detect_slots):
        """
        Vòng lặp ghi hình và gửi đến AI server để phát hiện.
        """
        last_detection_time = datetime.now()

        while self.is_running:
            try:
                if not self.cap or not self.cap.isOpened():
                    logger.error("Mất kết nối camera, đang thử kết nối lại...")
                    self.connect()
                    time.sleep(5)  # Chờ trước khi thử lại
                    continue

                # Đọc frame từ camera
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Không thể đọc frame, đang thử lại...")
                    time.sleep(0.5)
                    continue

                # Cập nhật frame mới nhất
                self.last_frame = frame
                self.last_frame_time = datetime.now()
                self.frame_count += 1

                # Kiểm tra xem đã đến lúc gửi phát hiện chưa
                current_time = datetime.now()
                time_diff = (current_time - last_detection_time).total_seconds()

                if time_diff >= interval:
                    # Gửi frame đến AI server để phát hiện
                    self.send_frame_for_detection(frame, detect_slots)
                    last_detection_time = current_time

                # Chờ một khoảng thời gian ngắn để giảm tải CPU
                time.sleep(0.05)

            except Exception as e:
                logger.error(f"Lỗi trong vòng lặp ghi hình: {e}")
                time.sleep(1)  # Chờ một chút trước khi thử lại

    def send_frame_for_detection(self, frame, detect_slots=True):
        """
        Gửi frame đến AI server để phát hiện.

        Args:
            frame (numpy.ndarray): Frame hình ảnh từ camera
            detect_slots (bool): Có phát hiện ô kệ hay không
        """
        try:
            # Chuyển đổi frame sang base64
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_image = base64.b64encode(buffer).decode('utf-8')

            # Chuẩn bị dữ liệu gửi đến AI server
            payload = {
                "image": encoded_image,
                "detect_slots": detect_slots,
                "camera_id": self.camera_id,
                "timestamp": datetime.now().isoformat()
            }

            # Gửi yêu cầu đến AI server
            logger.info(f"Đang gửi frame để phát hiện từ camera {self.camera_id}")
            response = requests.post(
                f"{self.ai_server_url}/detect",
                json=payload,
                timeout=30  # Tăng timeout vì phát hiện có thể mất thời gian
            )

            # Xử lý phản hồi
            if response.status_code == 200:
                result = response.json()
                num_detections = len(result.get("detections", []))
                logger.info(f"Phát hiện thành công: {num_detections} đối tượng được phát hiện")
                return result
            else:
                logger.error(f"Lỗi khi gửi frame để phát hiện: HTTP {response.status_code}")
                logger.error(f"Phản hồi: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Lỗi khi gửi frame để phát hiện: {e}")
            return None

    def get_current_frame(self):
        """Lấy frame hiện tại từ camera."""
        if self.last_frame is not None:
            return self.last_frame.copy()
        return None

    def get_status(self):
        """Lấy thông tin trạng thái của camera."""
        return {
            "camera_id": self.camera_id,
            "is_connected": self.is_connected,
            "is_running": self.is_running,
            "frame_count": self.frame_count,
            "last_frame_time": self.last_frame_time.isoformat() if self.last_frame_time else None,
            "rtsp_url": self.rtsp_url.replace(r"/\w+:\w+@/", "/***:***@/")  # Ẩn thông tin đăng nhập
        }


# Hàm chính để chạy camera IP
def run_ip_camera(rtsp_url, camera_id, username=None, password=None,
                  ai_server_url="http://localhost:5000", interval=1.0):
    """
    Chạy camera IP và gửi các frame để phát hiện.

    Args:
        rtsp_url (str): URL RTSP của camera IP
        camera_id (str): ID của camera
        username (str, optional): Tên đăng nhập nếu camera yêu cầu xác thực
        password (str, optional): Mật khẩu nếu camera yêu cầu xác thực
        ai_server_url (str): URL của AI server
        interval (float): Khoảng thời gian giữa các lần phát hiện (giây)
    """
    camera = IPCamera(camera_id, rtsp_url, username, password, ai_server_url)

    try:
        if camera.connect():
            camera.start_capture(interval=interval)

            # Giữ cho chương trình chạy
            while True:
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    break
        else:
            logger.error("Không thể kết nối đến camera IP.")

    except KeyboardInterrupt:
        logger.info("Dừng chương trình theo yêu cầu của người dùng.")
    finally:
        camera.disconnect()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kết nối camera IP và gửi hình ảnh để phát hiện")
    parser.add_argument("--rtsp", required=True, help="URL RTSP của camera IP")
    parser.add_argument("--id", default="ip_camera_1", help="ID của camera")
    parser.add_argument("--username", help="Tên đăng nhập camera (nếu cần)")
    parser.add_argument("--password", help="Mật khẩu camera (nếu cần)")
    parser.add_argument("--server", default="http://localhost:5000", help="URL của AI server")
    parser.add_argument("--interval", type=float, default=1.0, help="Khoảng thời gian giữa các lần phát hiện (giây)")

    args = parser.parse_args()

    run_ip_camera(
        rtsp_url=args.rtsp,
        camera_id=args.id,
        username=args.username,
        password=args.password,
        ai_server_url=args.server,
        interval=args.interval
    )