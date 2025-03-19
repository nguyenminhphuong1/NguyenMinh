import cv2
import time
import threading
import logging
import numpy as np
from queue import Queue
from typing import Dict, List, Optional, Tuple, Union

# Thiết lập logger
logger = logging.getLogger(__name__)


class Camera:
    """Lớp cơ sở cho tất cả các loại camera."""

    def __init__(self, camera_id: str, name: str, enabled: bool = True):
        self.camera_id = camera_id
        self.name = name
        self.enabled = enabled
        self.connected = False
        self.last_frame = None
        self.last_frame_time = 0
        self.frame_count = 0
        self.fps = 0
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None

    def connect(self) -> bool:
        """Kết nối đến camera. Cần được ghi đè trong lớp con."""
        raise NotImplementedError

    def disconnect(self) -> None:
        """Ngắt kết nối camera."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.connected = False
        logger.info(f"Đã ngắt kết nối camera {self.name} ({self.camera_id})")

    def get_frame(self) -> Optional[np.ndarray]:
        """Lấy khung hình mới nhất từ camera."""
        with self.lock:
            return self.last_frame.copy() if self.last_frame is not None else None

    def is_connected(self) -> bool:
        """Kiểm tra xem camera có kết nối hay không."""
        return self.connected

    def calculate_fps(self) -> None:
        """Tính toán số khung hình trên giây."""
        current_time = time.time()
        if self.last_frame_time == 0:
            self.last_frame_time = current_time
            return

        time_diff = current_time - self.last_frame_time
        if time_diff >= 1.0:  # Cập nhật FPS mỗi giây
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.last_frame_time = current_time
        else:
            self.frame_count += 1


class IPCamera(Camera):
    """Quản lý camera IP thông qua RTSP/HTTP stream."""

    def __init__(self, camera_id: str, name: str, url: str, resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30, reconnect_attempts: int = 3, timeout: int = 10, enabled: bool = True):
        super().__init__(camera_id, name, enabled)
        self.url = url
        self.resolution = resolution
        self.target_fps = fps
        self.reconnect_attempts = reconnect_attempts
        self.timeout = timeout
        self.cap = None

    def connect(self) -> bool:
        """Kết nối tới camera IP."""
        if not self.enabled:
            logger.warning(f"Camera {self.name} đã bị vô hiệu hóa.")
            return False

        for attempt in range(self.reconnect_attempts):
            try:
                logger.info(
                    f"Đang kết nối đến camera {self.name} ({self.camera_id}), lần thử {attempt + 1}/{self.reconnect_attempts}")

                # Sử dụng FFMPEG backend cho RTSP streams để cải thiện hiệu suất
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

                # Thiết lập độ phân giải
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

                # Kiểm tra kết nối
                if not self.cap.isOpened():
                    logger.error(f"Không thể mở kết nối tới camera {self.name}")
                    time.sleep(2)  # Chờ trước khi thử lại
                    continue

                # Đọc một khung hình để xác nhận kết nối
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.error(f"Không thể đọc khung hình từ camera {self.name}")
                    self.cap.release()
                    time.sleep(2)  # Chờ trước khi thử lại
                    continue

                # Kết nối thành công
                self.connected = True
                self._stop_event.clear()
                self._thread = threading.Thread(target=self._capture_loop, daemon=True)
                self._thread.start()
                logger.info(f"Đã kết nối thành công đến camera {self.name} ({self.camera_id})")
                return True

            except Exception as e:
                logger.error(f"Lỗi khi kết nối đến camera {self.name}: {str(e)}")
                if self.cap:
                    self.cap.release()
                time.sleep(2)  # Chờ trước khi thử lại

        logger.error(f"Không thể kết nối đến camera {self.name} sau {self.reconnect_attempts} lần thử")
        return False

    def _capture_loop(self) -> None:
        """Vòng lặp chính để liên tục lấy khung hình từ camera."""
        frame_interval = 1.0 / self.target_fps
        last_capture_time = 0

        while not self._stop_event.is_set():
            # Điều chỉnh tốc độ khung hình
            current_time = time.time()
            time_since_last_capture = current_time - last_capture_time

            if time_since_last_capture < frame_interval:
                time.sleep(frame_interval - time_since_last_capture)

            # Đọc khung hình
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    with self.lock:
                        self.last_frame = frame
                    self.calculate_fps()
                    last_capture_time = time.time()
                else:
                    logger.warning(f"Không thể đọc khung hình từ camera {self.name} ({self.camera_id})")
                    # Thử khởi động lại kết nối
                    self.cap.release()
                    if not self.connect():
                        # Nếu không thể kết nối lại, thoát khỏi vòng lặp
                        break
            else:
                logger.error(f"Kết nối đến camera {self.name} ({self.camera_id}) đã mất")
                # Thử khởi động lại kết nối
                if self.cap:
                    self.cap.release()
                if not self.connect():
                    # Nếu không thể kết nối lại, thoát khỏi vòng lặp
                    break

        # Dọn dẹp khi thoát vòng lặp
        if self.cap:
            self.cap.release()
        self.connected = False
        logger.info(f"Vòng lặp capture cho camera {self.name} ({self.camera_id}) đã kết thúc")


class IndustrialCamera(IPCamera):
    """Quản lý camera công nghiệp với các tùy chọn bổ sung."""

    def __init__(self, camera_id: str, name: str, url: str, resolution: Tuple[int, int] = (1920, 1080),
                 fps: int = 30, reconnect_attempts: int = 3, timeout: int = 15,
                 enabled: bool = True, advanced_settings: Dict = None):
        super().__init__(camera_id, name, url, resolution, fps, reconnect_attempts, timeout, enabled)
        self.advanced_settings = advanced_settings or {}

    def connect(self) -> bool:
        """Kết nối đến camera công nghiệp với các cài đặt nâng cao."""
        success = super().connect()

        if success and self.cap and self.advanced_settings:
            try:
                # Áp dụng các cài đặt nâng cao cho camera công nghiệp
                for setting, value in self.advanced_settings.items():
                    property_id = getattr(cv2, setting, None)
                    if property_id is not None:
                        self.cap.set(property_id, value)
                        logger.debug(f"Đã thiết lập {setting}={value} cho camera {self.name}")
            except Exception as e:
                logger.error(f"Lỗi khi áp dụng cài đặt nâng cao cho camera {self.name}: {str(e)}")

        return success


class WebcamCamera(Camera):
    """Quản lý webcam cục bộ kết nối qua USB."""

    def __init__(self, camera_id: str, name: str, device_id: int = 0,
                 resolution: Tuple[int, int] = (640, 480), fps: int = 30, enabled: bool = True):
        super().__init__(camera_id, name, enabled)
        self.device_id = device_id
        self.resolution = resolution
        self.target_fps = fps
        self.cap = None

    def connect(self) -> bool:
        """Kết nối đến webcam cục bộ."""
        if not self.enabled:
            logger.warning(f"Camera {self.name} đã bị vô hiệu hóa.")
            return False

        try:
            logger.info(f"Đang kết nối đến webcam {self.name} (ID thiết bị: {self.device_id})")

            self.cap = cv2.VideoCapture(self.device_id)

            # Thiết lập độ phân giải và FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Kiểm tra kết nối
            if not self.cap.isOpened():
                logger.error(f"Không thể mở kết nối tới webcam {self.name}")
                return False

            # Đọc một khung hình để xác nhận kết nối
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error(f"Không thể đọc khung hình từ webcam {self.name}")
                self.cap.release()
                return False

            # Kết nối thành công
            self.connected = True
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            logger.info(f"Đã kết nối thành công đến webcam {self.name}")
            return True

        except Exception as e:
            logger.error(f"Lỗi khi kết nối đến webcam {self.name}: {str(e)}")
            if self.cap:
                self.cap.release()
            return False

    def _capture_loop(self) -> None:
        """Vòng lặp chính để liên tục lấy khung hình từ webcam."""
        frame_interval = 1.0 / self.target_fps

        while not self._stop_event.is_set():
            start_time = time.time()

            # Đọc khung hình
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()

                if ret and frame is not None:
                    with self.lock:
                        self.last_frame = frame
                    self.calculate_fps()
                else:
                    logger.warning(f"Không thể đọc khung hình từ webcam {self.name}")
                    break
            else:
                logger.error(f"Kết nối đến webcam {self.name} đã mất")
                break

            # Điều chỉnh tốc độ khung hình
            elapsed_time = time.time() - start_time
            sleep_time = max(0, frame_interval - elapsed_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Dọn dẹp khi thoát vòng lặp
        if self.cap:
            self.cap.release()
        self.connected = False
        logger.info(f"Vòng lặp capture cho webcam {self.name} đã kết thúc")


class CameraManager:
    """Quản lý nhiều camera cùng một lúc."""

    def __init__(self, config: Dict):
        self.cameras = {}
        self.camera_groups = {}
        self.detection_zones = {}
        self.frame_buffer = {}
        self.frame_buffer_size = config.get("global_settings", {}).get("frame_buffer_size", 10)

        # Tạo các đối tượng camera từ cấu hình
        for camera_config in config.get("cameras", []):
            self._create_camera(camera_config)

        # Thiết lập nhóm camera
        for group in config.get("camera_groups", []):
            group_id = group.get("id")
            camera_ids = group.get("camera_ids", [])
            self.camera_groups[group_id] = {
                "name": group.get("name", group_id),
                "camera_ids": camera_ids
            }

        # Thiết lập khu vực phát hiện
        for zone in config.get("detection_zones", []):
            zone_id = f"{zone.get('camera_id')}_{zone.get('name')}"
            self.detection_zones[zone_id] = {
                "camera_id": zone.get("camera_id"),
                "name": zone.get("name"),
                "coordinates": zone.get("coordinates"),
                "detect_objects": zone.get("detect_objects", [])
            }

    def _create_camera(self, config: Dict) -> None:
        """Tạo đối tượng camera dựa trên cấu hình."""
        camera_id = config.get("id")
        camera_type = config.get("type", "ip")

        if camera_id in self.cameras:
            logger.warning(f"Camera có ID {camera_id} đã tồn tại, ghi đè lên cấu hình cũ")

        if camera_type == "ip":
            camera = IPCamera(
                camera_id=camera_id,
                name=config.get("name", f"Camera IP {camera_id}"),
                url=config.get("url"),
                resolution=tuple(config.get("resolution", (640, 480))),
                fps=config.get("fps", 30),
                reconnect_attempts=config.get("reconnect_attempts", 3),
                timeout=config.get("timeout", 10),
                enabled=config.get("enabled", True)
            )
        elif camera_type == "industrial":
            camera = IndustrialCamera(
                camera_id=camera_id,
                name=config.get("name", f"Camera công nghiệp {camera_id}"),
                url=config.get("url"),
                resolution=tuple(config.get("resolution", (1920, 1080))),
                fps=config.get("fps", 25),
                reconnect_attempts=config.get("reconnect_attempts", 3),
                timeout=config.get("timeout", 15),
                enabled=config.get("enabled", True),
                advanced_settings=config.get("advanced_settings")
            )
        elif camera_type == "webcam":
            camera = WebcamCamera(
                camera_id=camera_id,
                name=config.get("name", f"Webcam {camera_id}"),
                device_id=config.get("device_id", 0),
                resolution=tuple(config.get("resolution", (640, 480))),
                fps=config.get("fps", 30),
                enabled=config.get("enabled", True)
            )
        else:
            logger.error(f"Loại camera không được hỗ trợ: {camera_type}")
            return

        self.cameras[camera_id] = camera
        self.frame_buffer[camera_id] = Queue(maxsize=self.frame_buffer_size)
        logger.info(f"Đã tạo camera {camera.name} (ID: {camera_id}, loại: {camera_type})")

    def connect_all(self) -> Dict[str, bool]:
        """Kết nối tới tất cả các camera được bật."""
        results = {}

        for camera_id, camera in self.cameras.items():
            if camera.enabled:
                success = camera.connect()
                results[camera_id] = success
                if success:
                    logger.info(f"Đã kết nối thành công đến camera {camera.name} (ID: {camera_id})")
                else:
                    logger.error(f"Không thể kết nối đến camera {camera.name} (ID: {camera_id})")

        return results

    def disconnect_all(self) -> None:
        """Ngắt kết nối tất cả các camera."""
        for camera_id, camera in self.cameras.items():
            if camera.is_connected():
                camera.disconnect()
                logger.info(f"Đã ngắt kết nối camera {camera.name} (ID: {camera_id})")

    def get_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Lấy khung hình mới nhất từ camera cụ thể."""
        camera = self.cameras.get(camera_id)
        if camera and camera.is_connected():
            return camera.get_frame()
        return None

    def get_frames_from_group(self, group_id: str) -> Dict[str, np.ndarray]:
        """Lấy khung hình từ tất cả camera trong một nhóm."""
        frames = {}
        group = self.camera_groups.get(group_id)

        if group:
            for camera_id in group.get("camera_ids", []):
                frame = self.get_frame(camera_id)
                if frame is not None:
                    frames[camera_id] = frame

        return frames

    def apply_detection_zone(self, frame: np.ndarray, zone_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Áp dụng vùng phát hiện vào khung hình và trả về mặt nạ vùng."""
        if frame is None:
            return None, None

        zone = self.detection_zones.get(zone_id)
        if not zone:
            return frame, None

        # Tạo mặt nạ cho vùng phát hiện
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        coordinates = np.array(zone["coordinates"], dtype=np.int32)
        cv2.fillPoly(mask, [coordinates], 255)

        # Áp dụng mặt nạ vào khung hình để hiển thị vùng phát hiện
        frame_with_zone = frame.copy()
        cv2.polylines(frame_with_zone, [coordinates], True, (0, 255, 0), 2)

        return frame_with_zone, mask

    def update_frame_buffers(self) -> None:
        """Cập nhật bộ đệm khung hình cho tất cả camera đang hoạt động."""
        for camera_id, camera in self.cameras.items():
            if camera.is_connected():
                frame = camera.get_frame()
                if frame is not None:
                    # Thêm khung hình vào bộ đệm, loại bỏ khung hình cũ nhất nếu đầy
                    if self.frame_buffer[camera_id].full():
                        self.frame_buffer[camera_id].get()
                    self.frame_buffer[camera_id].put(frame)

    def get_camera_info(self, camera_id: str) -> Dict:
        """Lấy thông tin về camera cụ thể."""
        camera = self.cameras.get(camera_id)
        if not camera:
            return {}

        return {
            "id": camera.camera_id,
            "name": camera.name,
            "connected": camera.is_connected(),
            "enabled": camera.enabled,
            "fps": camera.fps,
            "resolution": camera.resolution if hasattr(camera, "resolution") else None
        }

    def get_all_cameras_info(self) -> List[Dict]:
        """Lấy thông tin về tất cả camera."""
        return [self.get_camera_info(camera_id) for camera_id in self.cameras]