import os
import sys
import time
import cv2
import numpy as np
import threading
import logging
import base64
import requests
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union

# Thêm đường dẫn gốc vào đường dẫn hệ thống
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QCheckBox, QTabWidget,
    QGroupBox, QGridLayout, QScrollArea, QSplitter, QFrame,
    QStatusBar, QMessageBox, QFileDialog, QSlider, QSpinBox
)
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont, QPalette
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QRect

from config.settings import UI_CONFIG, CAMERA_CONFIG, AI_SERVER
from utils.camera_manager import CameraManager
from utils.notification import NotificationManager

# Thiết lập logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gui")


class VideoProcessingThread(QThread):
    """Luồng riêng để xử lý video từ camera và phát hiện hàng hóa."""

    # Tín hiệu cập nhật hình ảnh
    updateFrame = pyqtSignal(np.ndarray, str)

    # Tín hiệu khi phát hiện hàng hóa
    detectionResult = pyqtSignal(bool, list, float, str)

    # Tín hiệu khi có lỗi
    errorSignal = pyqtSignal(str, str)

    def __init__(self, camera_id: str, camera_manager: CameraManager,
                 server_url: str, process_every_n_frames: int = 5):
        """
        Khởi tạo luồng xử lý video.

        Args:
            camera_id: ID của camera
            camera_manager: Đối tượng quản lý camera
            server_url: URL của server AI
            process_every_n_frames: Xử lý mỗi N khung hình
        """
        super().__init__()
        self.camera_id = camera_id
        self.camera_manager = camera_manager
        self.server_url = server_url
        self.process_every_n_frames = process_every_n_frames
        self.running = False
        self.processing = False
        self.frame_count = 0
        self.last_processing_time = 0
        self.detection_enabled = True

    def run(self):
        """Hàm chính của luồng xử lý video."""
        self.running = True

        while self.running:
            try:
                # Lấy khung hình từ camera
                frame = self.camera_manager.get_frame(self.camera_id)

                if frame is not None:
                    # Phát tín hiệu cập nhật hình ảnh
                    self.updateFrame.emit(frame, self.camera_id)

                    # Xử lý phát hiện đối tượng
                    if self.detection_enabled and self.frame_count % self.process_every_n_frames == 0:
                        if not self.processing:
                            self.processing = True

                            # Gửi khung hình đến server AI trong một luồng riêng
                            processing_thread = threading.Thread(
                                target=self._process_frame,
                                args=(frame.copy(),),
                                daemon=True
                            )
                            processing_thread.start()

                    self.frame_count += 1

                # Kiểm soát tốc độ khung hình
                time.sleep(0.03)  # Khoảng 30 FPS

            except Exception as e:
                logger.error(f"Lỗi trong luồng xử lý video ({self.camera_id}): {str(e)}")
                self.errorSignal.emit("Video Processing Error", f"Camera {self.camera_id}: {str(e)}")
                time.sleep(1)  # Chờ một chút trước khi thử lại

    def _process_frame(self, frame: np.ndarray) -> None:
        """
        Xử lý khung hình và gửi đến server AI.

        Args:
            frame: Khung hình cần xử lý
        """
        try:
            # Đảm bảo không quá thường xuyên gửi yêu cầu
            current_time = time.time()
            if current_time - self.last_processing_time < 0.2:  # Tối đa 5 yêu cầu/giây
                self.processing = False
                return

            self.last_processing_time = current_time

            # Chuẩn bị hình ảnh để gửi đến server
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')

            # Chuẩn bị dữ liệu
            payload = {
                "image": img_base64,
                "return_image": True,
                "camera_id": self.camera_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            }

            # Gửi yêu cầu đến server
            response = requests.post(
                f"{self.server_url}/detect",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()

                if result.get("success", False):
                    # Phát tín hiệu kết quả phát hiện
                    self.detectionResult.emit(
                        result.get("has_goods", False),
                        result.get("detections", []),
                        result.get("confidence", 0.0),
                        self.camera_id
                    )

                    # Lấy hình ảnh kết quả nếu có
                    if "result_image" in result:
                        result_img_data = base64.b64decode(result["result_image"])
                        nparr = np.frombuffer(result_img_data, np.uint8)
                        result_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if result_img is not None:
                            # Phát tín hiệu cập nhật hình ảnh kết quả
                            self.updateFrame.emit(result_img, self.camera_id)
                else:
                    logger.error(f"Lỗi từ server AI: {result.get('error', 'Unknown error')}")
            else:
                logger.error(f"Lỗi khi gửi yêu cầu đến server AI: HTTP {response.status_code}")

        except Exception as e:
            logger.error(f"Lỗi khi xử lý khung hình: {str(e)}")

        # Hoàn thành xử lý
        self.processing = False

    def stop(self):
        """Dừng luồng xử lý video."""
        self.running = False
        self.wait()

    def toggle_detection(self, enabled: bool):
        """Bật/tắt tính năng phát hiện."""
        self.detection_enabled = enabled


class VideoDisplay(QLabel):
    """Widget hiển thị video từ camera."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Đang chờ kết nối camera...")
        self.setStyleSheet("background-color: #222; color: white;")
        self.setScaledContents(True)

    def display_frame(self, frame: np.ndarray):
        """
        Hiển thị khung hình trên widget.

        Args:
            frame: Khung hình cần hiển thị (định dạng NumPy BGR)
        """
        if frame is None:
            return

        # Chuyển đổi từ BGR sang RGB và tạo QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Hiển thị hình ảnh
        self.setPixmap(QPixmap.fromImage(q_img))


class DetectionWidget(QWidget):
    """Widget hiển thị thông tin phát hiện."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.last_detection_time = 0
        self.has_goods = False
        self.confidence = 0.0
        self.camera_id = ""

    def init_ui(self):
        """Khởi tạo giao diện người dùng."""
        layout = QVBoxLayout(self)

        # Nhãn trạng thái
        self.status_label = QLabel("Trạng thái: Chưa phát hiện")
        self.status_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Nhãn độ tin cậy
        self.confidence_label = QLabel("Độ tin cậy: 0.0%")
        self.confidence_label.setFont(QFont("Arial", 12))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)

        # Nhãn thời gian phát hiện
        self.time_label = QLabel("Thời gian: --")
        self.time_label.setFont(QFont("Arial", 10))
        self.time_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.time_label)

        # Nhãn camera
        self.camera_label = QLabel("Camera: --")
        self.camera_label.setFont(QFont("Arial", 10))
        self.camera_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.camera_label)

        # Danh sách phát hiện
        self.detections_group = QGroupBox("Chi tiết phát hiện")
        detections_layout = QVBoxLayout()
        self.detections_label = QLabel("Chưa có phát hiện")
        self.detections_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.detections_label.setWordWrap(True)
        detections_layout.addWidget(self.detections_label)
        self.detections_group.setLayout(detections_layout)
        layout.addWidget(self.detections_group)

        layout.addStretch()

    def update_detection(self, has_goods: bool, detections: List[Dict], confidence: float, camera_id: str):
        """
        Cập nhật thông tin phát hiện.

        Args:
            has_goods: True nếu phát hiện hàng hóa, False nếu không
            detections: Danh sách các phát hiện
            confidence: Độ tin cậy của phát hiện
            camera_id: ID của camera phát hiện
        """
        self.has_goods = has_goods
        self.confidence = confidence
        self.camera_id = camera_id
        self.last_detection_time = time.time()

        # Cập nhật trạng thái
        if has_goods:
            self.status_label.setText("Trạng thái: ĐÃ PHÁT HIỆN HÀNG HÓA")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setText("Trạng thái: KHÔNG CÓ HÀNG HÓA")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

        # Cập nhật độ tin cậy
        self.confidence_label.setText(f"Độ tin cậy: {confidence:.2%}")

        # Cập nhật thời gian
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self.time_label.setText(f"Thời gian: {current_time}")

        # Cập nhật camera
        self.camera_label.setText(f"Camera: {camera_id}")

        # Cập nhật danh sách phát hiện
        if detections:
            detection_text = ""
            for i, det in enumerate(detections):
                cls = det.get("class", "unknown")
                conf = det.get("confidence", 0.0)
                bbox = det.get("bbox", [0, 0, 0, 0])
                detection_text += f"{i + 1}. {cls.capitalize()}: {conf:.2%} - Vị trí: {bbox}\n"
            self.detections_label.setText(detection_text)
        else:
            self.detections_label.setText("Không có chi tiết phát hiện")


class MainWindow(QMainWindow):
    """Giao diện chính của ứng dụng phát hiện hàng hóa."""

    def __init__(self, config: Dict = None):
        super().__init__()
        self.config = config or {}
        self.ui_config = self.config.get("ui_config", UI_CONFIG)
        self.camera_config = self.config.get("camera_config", CAMERA_CONFIG)
        self.ai_server_config = self.config.get("ai_server", AI_SERVER)

        # Khởi tạo các thành phần
        self.init_managers()
        self.init_ui()
        self.init_connections()

        # Bắt đầu cập nhật UI
        self.start_update_timer()

    def init_managers(self):
        """Khởi tạo các đối tượng quản lý."""
        # Khởi tạo quản lý camera
        self.camera_manager = CameraManager(self.camera_config)

        # Khởi tạo quản lý thông báo
        self.notification_manager = NotificationManager(
            self.config.get("notification_server", {
                "url": "http://localhost:8000/notification",
                "api_key": "your_api_key_here",
                "retry_attempts": 3,
                "timeout": 5
            })
        )

        # Khởi tạo luồng xử lý video
        self.video_threads = {}

    def init_ui(self):
        """Khởi tạo giao diện người dùng."""
        # Thiết lập cửa sổ chính
        self.setWindowTitle(self.ui_config.get("window_title", "Hệ thống phát hiện hàng hóa"))
        self.setMinimumSize(1024, 768)
        self.resize(
            self.ui_config.get("window_width", 1280),
            self.ui_config.get("window_height", 720)
        )

        # Widget chính
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Tạo thanh công cụ
        toolbar_layout = QHBoxLayout()

        # Nút kết nối camera
        self.connect_button = QPushButton("Kết nối tất cả camera")
        self.connect_button.setToolTip("Kết nối đến tất cả camera đã cấu hình")
        toolbar_layout.addWidget(self.connect_button)

        # Nút ngắt kết nối camera
        self.disconnect_button = QPushButton("Ngắt kết nối")
        self.disconnect_button.setToolTip("Ngắt kết nối tất cả camera")
        toolbar_layout.addWidget(self.disconnect_button)

        # Combobox chọn camera
        self.camera_combo = QComboBox()
        self.camera_combo.setToolTip("Chọn camera để hiển thị")
        toolbar_layout.addWidget(QLabel("Camera:"))
        toolbar_layout.addWidget(self.camera_combo)

        # Checkbox bật/tắt phát hiện
        self.detection_checkbox = QCheckBox("Bật phát hiện")
        self.detection_checkbox.setChecked(True)
        toolbar_layout.addWidget(self.detection_checkbox)

        # Spinbox tần suất phát hiện
        toolbar_layout.addWidget(QLabel("Tần suất phát hiện:"))
        self.detection_freq_spinbox = QSpinBox()
        self.detection_freq_spinbox.setRange(1, 30)
        self.detection_freq_spinbox.setValue(5)
        self.detection_freq_spinbox.setSuffix(" khung hình")
        toolbar_layout.addWidget(self.detection_freq_spinbox)

        # Nút cài đặt
        self.settings_button = QPushButton("Cài đặt")
        toolbar_layout.addWidget(self.settings_button)

        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        # Tạo splitter để chia màn hình
        splitter = QSplitter(Qt.Horizontal)

        # Khu vực hiển thị video
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)

        # Nhãn tiêu đề video
        video_title = QLabel("Hình ảnh Camera")
        video_title.setAlignment(Qt.AlignCenter)
        video_title.setFont(QFont("Arial", 12, QFont.Bold))
        video_layout.addWidget(video_title)

        # Widget hiển thị video
        self.video_display = VideoDisplay()
        video_layout.addWidget(self.video_display)

        splitter.addWidget(video_widget)

        # Khu vực hiển thị kết quả phát hiện
        detection_widget = QWidget()
        detection_layout = QVBoxLayout(detection_widget)

        # Nhãn tiêu đề phát hiện
        detection_title = QLabel("Kết quả phát hiện")
        detection_title.setAlignment(Qt.AlignCenter)
        detection_title.setFont(QFont("Arial", 12, QFont.Bold))
        detection_layout.addWidget(detection_title)

        # Widget phát hiện
        self.detection_display = DetectionWidget()
        detection_layout.addWidget(self.detection_display)

        splitter.addWidget(detection_widget)

        # Thiết lập tỉ lệ mặc định cho splitter
        splitter.setSizes([700, 300])

        main_layout.addWidget(splitter)

        # Thanh trạng thái
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Sẵn sàng")

        # Cập nhật danh sách camera
        self.update_camera_list()

    def init_connections(self):
        """Kết nối các tín hiệu và khe."""
        # Kết nối nút kết nối camera
        self.connect_button.clicked.connect(self.connect_cameras)

        # Kết nối nút ngắt kết nối camera
        self.disconnect_button.clicked.connect(self.disconnect_cameras)

        # Kết nối combobox chọn camera
        self.camera_combo.currentIndexChanged.connect(self.change_camera)

        # Kết nối checkbox phát hiện
        self.detection_checkbox.stateChanged.connect(self.toggle_detection)

        # Kết nối spinbox tần suất phát hiện
        self.detection_freq_spinbox.valueChanged.connect(self.change_detection_frequency)

        # Kết nối nút cài đặt
        self.settings_button.clicked.connect(self.show_settings)

    def start_update_timer(self):
        """Khởi động timer cập nhật UI."""
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(self.ui_config.get("update_interval", 100))

    def update_camera_list(self):
        """Cập nhật danh sách camera trong combobox."""
        self.camera_combo.clear()

        for camera_info in self.camera_manager.get_all_cameras_info():
            if camera_info.get("enabled", True):
                camera_id = camera_info.get("id", "")
                camera_name = camera_info.get("name", camera_id)
                self.camera_combo.addItem(camera_name, camera_id)

    def connect_cameras(self):
        """Kết nối đến tất cả camera."""
        self.status_bar.showMessage("Đang kết nối camera...")

        # Ngắt kết nối các luồng hiện tại nếu có
        self.disconnect_cameras()

        # Kết nối tất cả camera
        results = self.camera_manager.connect_all()

        connected_count = sum(1 for success in results.values() if success)
        total_count = len(results)

        if connected_count > 0:
            self.status_bar.showMessage(
                f"Đã kết nối {connected_count}/{total_count} camera"
            )

            # Khởi động luồng xử lý video cho mỗi camera
            for camera_id, success in results.items():
                if success:
                    self.start_video_thread(camera_id)

            # Chọn camera đầu tiên đã kết nối
            if self.camera_combo.count() > 0:
                self.camera_combo.setCurrentIndex(0)
                self.change_camera(0)
        else:
            self.status_bar.showMessage("Không thể kết nối đến bất kỳ camera nào")
            QMessageBox.warning(
                self,
                "Lỗi kết nối",
                "Không thể kết nối đến bất kỳ camera nào. Vui lòng kiểm tra cấu hình camera."
            )

    def disconnect_cameras(self):
        """Ngắt kết nối tất cả camera."""
        # Dừng tất cả luồng xử lý video
        for thread in self.video_threads.values():
            thread.stop()

        self.video_threads.clear()

        # Ngắt kết nối tất cả camera
        self.camera_manager.disconnect_all()

        # Đặt lại hiển thị video
        self.video_display.setText("Đang chờ kết nối camera...")
        self.status_bar.showMessage("Đã ngắt kết nối tất cả camera")

    def start_video_thread(self, camera_id):
        """
        Khởi động luồng xử lý video cho camera.

        Args:
            camera_id: ID của camera
        """
        if camera_id in self.video_threads:
            # Dừng luồng cũ nếu đang chạy
            self.video_threads[camera_id].stop()

        # Tạo luồng mới
        thread = VideoProcessingThread(
            camera_id=camera_id,
            camera_manager=self.camera_manager,
            server_url=f"http://{self.ai_server_config.get('host')}:{self.ai_server_config.get('port')}",
            process_every_n_frames=self.detection_freq_spinbox.value()
        )

        # Kết nối tín hiệu
        thread.updateFrame.connect(self.update_frame)
        thread.detectionResult.connect(self.handle_detection_result)
        thread.errorSignal.connect(self.handle_error)

        # Khởi động luồng
        thread.start()

        # Lưu luồng
        self.video_threads[camera_id] = thread

    def change_camera(self, index):
        """
        Thay đổi camera đang hiển thị.

        Args:
            index: Chỉ số của camera trong combobox
        """
        if index < 0:
            return

        camera_id = self.camera_combo.itemData(index)

        if not camera_id:
            return

        # Kiểm tra xem camera đã được kết nối chưa
        camera_info = self.camera_manager.get_camera_info(camera_id)

        if not camera_info.get("connected", False):
            # Thử kết nối camera
            self.camera_manager.cameras[camera_id].connect()

            # Khởi động luồng xử lý video nếu chưa có
            if camera_id not in self.video_threads:
                self.start_video_thread(camera_id)

        self.status_bar.showMessage(f"Đang hiển thị camera: {self.camera_combo.currentText()}")

    def update_frame(self, frame, camera_id):
        """
        Cập nhật khung hình hiển thị.

        Args:
            frame: Khung hình mới
            camera_id: ID của camera gửi khung hình
        """
        # Kiểm tra xem khung hình có phải từ camera đang hiển thị không
        current_camera_id = self.camera_combo.currentData()

        if current_camera_id == camera_id:
            self.video_display.display_frame(frame)

    def handle_detection_result(self, has_goods, detections, confidence, camera_id):
        """
        Xử lý kết quả phát hiện.

        Args:
            has_goods: True nếu phát hiện hàng hóa, False nếu không
            detections: Danh sách các phát hiện
            confidence: Độ tin cậy của phát hiện
            camera_id: ID của camera phát hiện
        """
        # Cập nhật màn hình hiển thị phát hiện
        self.detection_display.update_detection(has_goods, detections, confidence, camera_id)

        # Gửi thông báo đến server
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        self.notification_manager.send_goods_detection_notification(
            camera_id=camera_id,
            has_goods=has_goods,
            confidence=confidence,
            timestamp=timestamp,
            detection_data=detections
        )

    def handle_error(self, error_type, message):
        """
        Xử lý lỗi từ luồng xử lý video.

        Args:
            error_type: Loại lỗi
            message: Thông điệp lỗi
        """
        logger.error(f"{error_type}: {message}")
        self.status_bar.showMessage(f"Lỗi: {message}", 5000)

        # Gửi thông báo lỗi
        self.notification_manager.send_error_notification(
            error_type=error_type,
            message=message,
            source="GUI"
        )

    def toggle_detection(self, state):
        """
        Bật/tắt tính năng phát hiện.

        Args:
            state: Trạng thái của checkbox
        """
        enabled = state == Qt.Checked

        for thread in self.video_threads.values():
            thread.toggle_detection(enabled)

        if enabled:
            self.status_bar.showMessage("Đã bật tính năng phát hiện hàng hóa", 2000)
        else:
            self.status_bar.showMessage("Đã tắt tính năng phát hiện hàng hóa", 2000)

    def change_detection_frequency(self, value):
        """
        Thay đổi tần suất phát hiện.

        Args:
            value: Giá trị mới
        """
        for thread in self.video_threads.values():
            thread.process_every_n_frames = value

        self.status_bar.showMessage(f"Đã thay đổi tần suất phát hiện: Mỗi {value} khung hình", 2000)

    def show_settings(self):
        """Hiển thị hộp thoại cài đặt."""
        # Tạo hộp thoại cài đặt đơn giản
        QMessageBox.information(
            self,
            "Cài đặt",
            "Tính năng cài đặt đang được phát triển."
        )

    def update_ui(self):
        """Cập nhật UI định kỳ."""
        # Cập nhật trạng thái camera
        for i in range(self.camera_combo.count()):
            camera_id = self.camera_combo.itemData(i)
            camera_info = self.camera_manager.get_camera_info(camera_id)

            if camera_info:
                connected = camera_info.get("connected", False)
                item_text = camera_info.get("name", camera_id)

                if connected:
                    fps = camera_info.get("fps", 0)
                    item_text = f"{item_text} ({fps:.1f} FPS)"
                else:
                    item_text = f"{item_text} (Đã ngắt kết nối)"

                self.camera_combo.setItemText(i, item_text)

    def closeEvent(self, event):
        """Xử lý sự kiện đóng cửa sổ."""
        # Ngắt kết nối tất cả camera
        self.disconnect_cameras()

        # Dừng quản lý thông báo
        self.notification_manager.stop()

        # Chấp nhận sự kiện
        event.accept()