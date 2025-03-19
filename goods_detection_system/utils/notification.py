import requests
import json
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
import threading

# Thiết lập logger
logger = logging.getLogger(__name__)


class NotificationManager:
    """Quản lý gửi thông báo đến server khác."""

    def __init__(self, config: Dict):
        self.url = config.get("url", "http://localhost:8000/notification")
        self.api_key = config.get("api_key", "")
        self.retry_attempts = config.get("retry_attempts", 3)
        self.timeout = config.get("timeout", 5)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.last_status = {}  # Lưu trạng thái cuối cùng để tránh gửi thông báo trùng lặp
        self._queue = []  # Hàng đợi thông báo
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._start_worker()

    def _start_worker(self):
        """Khởi động luồng worker để xử lý hàng đợi thông báo."""
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def _process_queue(self):
        """Xử lý hàng đợi thông báo trong nền."""
        while not self._stop_event.is_set():
            # Đợi nếu hàng đợi trống
            if not self._queue:
                time.sleep(0.1)
                continue

            # Lấy thông báo từ hàng đợi
            with self._lock:
                if self._queue:
                    notification = self._queue.pop(0)
                else:
                    continue

            # Gửi thông báo
            self._send_notification_with_retry(notification)

    def stop(self):
        """Dừng luồng worker."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)

    def send_goods_detection_notification(self, camera_id: str, has_goods: bool,
                                          confidence: float, timestamp: str,
                                          detection_data: List[Dict] = None) -> bool:
        """
        Gửi thông báo phát hiện hàng hóa.

        Args:
            camera_id: ID của camera phát hiện
            has_goods: True nếu phát hiện hàng hóa, False nếu không
            confidence: Độ tin cậy của phát hiện
            timestamp: Dấu thời gian của phát hiện
            detection_data: Dữ liệu phát hiện chi tiết (tùy chọn)

        Returns:
            bool: True nếu thông báo được thêm vào hàng đợi thành công
        """
        # Kiểm tra xem trạng thái đã thay đổi hay chưa
        current_status = (camera_id, has_goods)
        last_detection = self.last_status.get(camera_id)

        # Nếu trạng thái không thay đổi và khoảng cách giữa các thông báo quá nhỏ, bỏ qua
        if last_detection:
            last_has_goods, last_time = last_detection
            if last_has_goods == has_goods and time.time() - last_time < 1.0:
                return True

        # Cập nhật trạng thái cuối cùng
        self.last_status[camera_id] = (has_goods, time.time())

        # Tạo dữ liệu thông báo
        notification = {
            "type": "goods_detection",
            "camera_id": camera_id,
            "has_goods": has_goods,
            "confidence": confidence,
            "timestamp": timestamp,
            "detection_data": detection_data or []
        }

        # Thêm thông báo vào hàng đợi
        with self._lock:
            self._queue.append(notification)

        return True

    # Phương thức mới để gửi thông báo trạng thái ô kệ
    def send_slots_status(self, slots_data: Dict) -> bool:
        """
        Gửi thông báo về trạng thái các ô kệ.

        Args:
            slots_data: Dữ liệu về trạng thái các ô kệ
                {
                    "timestamp": "YYYY-MM-DD HH:MM:SS",
                    "camera_id": "camera_id",
                    "slots_status": [
                        {
                            "slot_id": 1,
                            "has_goods": True/False,
                            "position": {"row": 0, "column": 0},
                            "confidence": 0.95
                        },
                        ...
                    ]
                }

        Returns:
            bool: True nếu thông báo được thêm vào hàng đợi thành công
        """
        # Tạo thông báo trạng thái ô kệ
        notification = {
            "type": "slots_status",
            "timestamp": slots_data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
            "camera_id": slots_data.get("camera_id", "unknown"),
            "slots_data": slots_data.get("slots_status", [])
        }

        # Log thông tin
        logger.info(
            f"Gửi thông báo trạng thái {len(slots_data.get('slots_status', []))} ô kệ từ camera {slots_data.get('camera_id', 'unknown')}")

        # Thêm thông báo vào hàng đợi
        with self._lock:
            self._queue.append(notification)

        return True

    def _send_notification_with_retry(self, data: Dict) -> Tuple[bool, Optional[Dict]]:
        """Gửi thông báo với cơ chế thử lại."""
        for attempt in range(self.retry_attempts):
            try:
                response = requests.post(
                    self.url,
                    headers=self.headers,
                    json=data,
                    timeout=self.timeout
                )

                if response.status_code in (200, 201, 202):
                    logger.debug(f"Gửi thông báo thành công: {data.get('type')} - {response.status_code}")
                    try:
                        return True, response.json()
                    except json.JSONDecodeError:
                        return True, {"status": "success", "raw_response": response.text}
                else:
                    logger.warning(
                        f"Lỗi khi gửi thông báo (lần thử {attempt + 1}/{self.retry_attempts}): "
                        f"HTTP {response.status_code} - {response.text}"
                    )

                    # Đợi trước khi thử lại
                    if attempt < self.retry_attempts - 1:
                        time.sleep(0.5 * (attempt + 1))

            except requests.RequestException as e:
                logger.error(
                    f"Lỗi mạng khi gửi thông báo (lần thử {attempt + 1}/{self.retry_attempts}): {str(e)}"
                )

                # Đợi trước khi thử lại
                if attempt < self.retry_attempts - 1:
                    time.sleep(0.5 * (attempt + 1))

        logger.error(f"Không thể gửi thông báo sau {self.retry_attempts} lần thử: {data.get('type')}")
        return False, None

    def send_error_notification(self, error_type: str, message: str, source: str = None) -> bool:
        """
        Gửi thông báo lỗi.

        Args:
            error_type: Loại lỗi
            message: Thông điệp lỗi
            source: Nguồn gốc lỗi (tùy chọn)

        Returns:
            bool: True nếu thông báo được thêm vào hàng đợi thành công
        """
        notification = {
            "type": "error",
            "error_type": error_type,
            "message": message,
            "source": source or "unknown",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with self._lock:
            self._queue.append(notification)

        return True

    def send_system_status(self, status: Dict[str, Any]) -> bool:
        """
        Gửi trạng thái hệ thống.

        Args:
            status: Dict chứa thông tin trạng thái hệ thống

        Returns:
            bool: True nếu thông báo được thêm vào hàng đợi thành công
        """
        notification = {
            "type": "system_status",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": status
        }

        with self._lock:
            self._queue.append(notification)

        return True

    def send_camera_status(self, camera_id: str, is_connected: bool, fps: float = None, error: str = None) -> bool:
        """
        Gửi trạng thái camera.

        Args:
            camera_id: ID của camera
            is_connected: True nếu camera đang kết nối, False nếu không
            fps: Số khung hình trên giây (tùy chọn)
            error: Thông điệp lỗi nếu có (tùy chọn)

        Returns:
            bool: True nếu thông báo được thêm vào hàng đợi thành công
        """
        notification = {
            "type": "camera_status",
            "camera_id": camera_id,
            "is_connected": is_connected,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        if fps is not None:
            notification["fps"] = fps

        if error:
            notification["error"] = error

        with self._lock:
            self._queue.append(notification)

        return True