import os
import time
import json
import logging
import numpy as np
import base64
import cv2
import threading
from typing import Dict, List, Tuple, Optional, Union
from flask import Flask, request, jsonify, Response, current_app

# Thêm thư mục gốc vào đường dẫn
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fast_rcnn_model import FastRCNNModel
from models.slot_detection_model import SlotDetectionModel  # Import mô hình phát hiện ô kệ
from utils.image_processor import ImageProcessor
from utils.notification import NotificationManager  # Import quản lý thông báo
from config.settings import AI_SERVER, NOTIFICATION_SERVER
from server.api.endpoints import register_endpoints
from server.api.slot_detection_endpoints import register_slot_detection_endpoints

# Thiết lập logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ai_server")


class AIServer:
    """Server AI xử lý các yêu cầu phát hiện đối tượng."""

    def __init__(self, config: Dict = None):
        """
        Khởi tạo server AI.

        Args:
            config: Cấu hình server
        """
        self.config = config or AI_SERVER
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 5000)
        self.model_path = self.config.get("model_path")
        self.threshold = self.config.get("threshold", 0.7)
        self.device = self.config.get("device", "cpu")

        # Khởi tạo Flask app
        self.app = Flask(__name__)

        # Khởi tạo bộ xử lý hình ảnh
        self.image_processor = ImageProcessor({
            "resize_dimensions": (800, 800),
            "normalize": True,
            "use_gpu": self.device == "cuda"
        })

        # Khởi tạo quản lý thông báo
        self.notification_manager = NotificationManager(
            self.config.get("notification_server", NOTIFICATION_SERVER)
        )

        # Khởi tạo mô hình Fast R-CNN
        logger.info(f"Đang tải mô hình từ {self.model_path}")
        self.model = FastRCNNModel(
            model_path=self.model_path,
            num_classes=6,  # background + 5 loại hàng hóa
            device=self.device,
            confidence_threshold=self.threshold
        )

        # Khởi tạo mô hình phát hiện ô kệ
        self.slot_model = SlotDetectionModel(
            model_path=self.model_path,
            device=self.device,
            confidence_threshold=self.threshold
        )

        # Thiết lập routes sau khi mô hình đã được khởi tạo
        self._setup_routes()

        # Thống kê
        self.stats = {
            "total_requests": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "avg_processing_time": 0.0,
            "start_time": time.time()
        }
        self.stats_lock = threading.Lock()

    def _setup_routes(self):
        """Thiết lập các route cho API."""
        # Đăng ký các endpoint API chuẩn
        register_endpoints(self.app, self.model, self.image_processor)

        # Đăng ký endpoint mới cho phát hiện ô kệ
        register_slot_detection_endpoints(self.app, self.model_path, self.device)

        # Thêm notification manager vào app config để sử dụng trong các endpoint
        self.app.config['notification_manager'] = self.notification_manager

        @self.app.route('/detect', methods=['POST'])
        def detect():
            """Endpoint chính để phát hiện hàng hóa trong hình ảnh."""
            start_time = time.time()

            try:
                # Đọc dữ liệu từ request
                data = request.json

                if not data:
                    return jsonify({
                        "success": False,
                        "error": "Không có dữ liệu JSON"
                    }), 400

                # Lấy hình ảnh từ mã hóa base64
                image_b64 = data.get("image")
                if not image_b64:
                    return jsonify({
                        "success": False,
                        "error": "Không tìm thấy trường 'image'"
                    }), 400

                # Giải mã hình ảnh
                try:
                    # Loại bỏ tiền tố data URL nếu có
                    if "base64," in image_b64:
                        image_b64 = image_b64.split("base64,")[1]

                    image_data = base64.b64decode(image_b64)
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if image is None:
                        raise ValueError("Không thể giải mã hình ảnh")
                except Exception as e:
                    return jsonify({
                        "success": False,
                        "error": f"Lỗi khi giải mã hình ảnh: {str(e)}"
                    }), 400

                # Kiểm tra yêu cầu phát hiện ô kệ
                detect_slots = data.get("detect_slots", False)
                camera_id = data.get("camera_id", "unknown")

                if detect_slots:
                    # Sử dụng mô hình phát hiện ô kệ
                    has_goods, detections, overall_confidence = self.slot_model.predict(image)

                    # Tạo hình ảnh kết quả
                    _, result_image = self.slot_model.detect_and_classify_slots(image)

                    # Gửi thông báo về trạng thái các ô kệ
                    self._send_slots_notification(camera_id, detections)

                else:
                    # Tiền xử lý hình ảnh
                    processed_image = self.image_processor.preprocess(image)

                    # Phát hiện hàng hóa
                    has_goods, detections, overall_confidence = self.model.detect_goods(processed_image)

                    # Tạo hình ảnh kết quả
                    result_image = self.image_processor.draw_detection_results(
                        image, detections, confidence_threshold=self.threshold
                    )

                    # Gửi thông báo phát hiện hàng hóa thông thường
                    self.notification_manager.send_goods_detection_notification(
                        camera_id=camera_id,
                        has_goods=has_goods,
                        confidence=overall_confidence,
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        detection_data=detections
                    )

                # Thêm dấu thời gian
                result_image = self.image_processor.add_timestamp(result_image)

                # Mã hóa hình ảnh kết quả
                _, buffer = cv2.imencode('.jpg', result_image)
                result_image_b64 = base64.b64encode(buffer).decode('utf-8')

                # Cập nhật thống kê
                processing_time = time.time() - start_time
                with self.stats_lock:
                    self.stats["total_requests"] += 1
                    if has_goods:
                        self.stats["successful_detections"] += 1
                    else:
                        self.stats["failed_detections"] += 1

                    # Cập nhật thời gian xử lý trung bình
                    self.stats["avg_processing_time"] = (
                            (self.stats["avg_processing_time"] * (self.stats["total_requests"] - 1) + processing_time) /
                            self.stats["total_requests"]
                    )

                # Trả về kết quả
                response = {
                    "success": True,
                    "has_goods": has_goods,
                    "confidence": overall_confidence,
                    "detections": detections,
                    "processing_time": processing_time,
                    "result_image": result_image_b64
                }

                # Nếu là phát hiện ô kệ, thêm thông tin tổng kết
                if detect_slots:
                    empty_count = sum(1 for d in detections if d["class"] == "empty_slot")
                    filled_count = sum(1 for d in detections if d["class"] == "filled_slot")
                    total_count = len(detections)

                    response["summary"] = {
                        "total_slots": total_count,
                        "empty_slots": empty_count,
                        "filled_slots": filled_count,
                        "occupancy_rate": filled_count / total_count if total_count > 0 else 0
                    }

                return jsonify(response)

            except Exception as e:
                logger.error(f"Lỗi khi xử lý yêu cầu phát hiện: {str(e)}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500

        # Thêm endpoint mới để gửi thông báo thủ công
        @self.app.route('/api/send_notification', methods=['POST'])
        def send_notification():
            """Endpoint để gửi thông báo thủ công về trạng thái ô kệ."""
            try:
                data = request.json
                if not data:
                    return jsonify({"success": False, "error": "Không có dữ liệu JSON"}), 400

                notification_type = data.get("type")
                notification_data = data.get("data")

                if not notification_type or not notification_data:
                    return jsonify({
                        "success": False,
                        "error": "Thiếu thông tin 'type' hoặc 'data'"
                    }), 400

                # Xử lý các loại thông báo khác nhau
                if notification_type == "slots_status":
                    success = self.notification_manager.send_slots_status(notification_data)
                elif notification_type == "goods_detection":
                    success = self.notification_manager.send_goods_detection_notification(
                        camera_id=notification_data.get("camera_id", "unknown"),
                        has_goods=notification_data.get("has_goods", False),
                        confidence=notification_data.get("confidence", 0.0),
                        timestamp=notification_data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S")),
                        detection_data=notification_data.get("detection_data", [])
                    )
                else:
                    return jsonify({
                        "success": False,
                        "error": f"Loại thông báo không được hỗ trợ: {notification_type}"
                    }), 400

                return jsonify({
                    "success": success,
                    "message": f"Đã thêm thông báo loại '{notification_type}' vào hàng đợi"
                })

            except Exception as e:
                logger.error(f"Lỗi khi xử lý yêu cầu gửi thông báo: {str(e)}")
                return jsonify({
                    "success": False,
                    "error": str(e)
                }), 500

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Endpoint kiểm tra trạng thái hoạt động của server."""
            with self.stats_lock:
                uptime = time.time() - self.stats["start_time"]

                return jsonify({
                    "status": "ok",
                    "version": "1.0.0",
                    "uptime": uptime,
                    "stats": {
                        "total_requests": self.stats["total_requests"],
                        "successful_detections": self.stats["successful_detections"],
                        "failed_detections": self.stats["failed_detections"],
                        "avg_processing_time": self.stats["avg_processing_time"]
                    },
                    "device": self.device
                })

    def _send_slots_notification(self, camera_id: str, detections: List[Dict]):
        """
        Gửi thông báo về trạng thái các ô kệ.

        Args:
            camera_id: ID của camera
            detections: Danh sách các phát hiện ô kệ
        """
        # Chuẩn bị dữ liệu thông báo
        slots_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_id": camera_id,
            "slots_status": []
        }

        # Thêm thông tin từng ô kệ
        for detection in detections:
            slot_id = detection.get("slot_id", 0)
            is_filled = detection.get("class") == "filled_slot"
            row = detection.get("row", 0)
            column = detection.get("column", 0)

            slot_info = {
                "slot_id": slot_id,
                "has_goods": is_filled,
                "position": {"row": row, "column": column},
                "confidence": detection.get("confidence", 0.0)
            }
            slots_data["slots_status"].append(slot_info)

        # Gửi thông báo qua NotificationManager
        if slots_data["slots_status"]:
            self.notification_manager.send_slots_status(slots_data)
            logger.info(f"Đã gửi thông báo trạng thái {len(slots_data['slots_status'])} ô kệ từ camera {camera_id}")

    def start(self, debug: bool = False, threaded: bool = True):
        """
        Khởi động server AI.

        Args:
            debug: Chế độ debug
            threaded: Sử dụng đa luồng
        """
        logger.info(f"Khởi động server AI tại http://{self.host}:{self.port}")
        self.app.run(
            host=self.host,
            port=self.port,
            debug=debug,
            threaded=threaded
        )

    def stop(self):
        """Dừng server và giải phóng tài nguyên."""
        # Dừng notification manager
        self.notification_manager.stop()
        logger.info("Đã dừng notification manager")


# Hàm khởi động server
def run_server(config: Dict = None):
    """Hàm tiện ích để khởi động server từ dòng lệnh."""
    server = AIServer(config)
    try:
        server.start()
    finally:
        server.stop()


if __name__ == "__main__":
    run_server()