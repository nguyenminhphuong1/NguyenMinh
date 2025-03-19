from flask import Blueprint, request, jsonify, Response, current_app
import os
import time
import json
import base64
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union

# Thêm thư mục gốc vào đường dẫn
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import các module cần thiết
from models.slot_detection_model import SlotDetectionModel
from utils.image_processor import ImageProcessor

# Thiết lập logger
logger = logging.getLogger(__name__)

# Tạo blueprint
slot_detection_bp = Blueprint('slot_detection', __name__)

# Biến toàn cục để theo dõi việc khởi tạo
_global_slot_model = None
_model_initialization_error = None


@slot_detection_bp.route('/detect_slots', methods=['POST'])
def detect_slots():
    """API endpoint để phát hiện trạng thái các ô kệ trong hình ảnh."""
    start_time = time.time()

    # Sử dụng biến toàn cục
    global _global_slot_model

    # Lấy mô hình từ ứng dụng Flask hoặc biến toàn cục
    slot_model = current_app.config.get('slot_model') or _global_slot_model

    # Nếu mô hình chưa được khởi tạo, tạo nó
    if not slot_model:
        try:
            model_path = current_app.config.get('model_path')
            device = current_app.config.get('device', 'cpu')
            logger.info(f"Đang khởi tạo mô hình phát hiện ô kệ từ {model_path}")
            slot_model = SlotDetectionModel(model_path=model_path, device=device)

            # Lưu mô hình vào cả config và biến toàn cục
            current_app.config['slot_model'] = slot_model
            _global_slot_model = slot_model
            logger.info("Đã khởi tạo mô hình phát hiện ô kệ thành công")
        except Exception as e:
            error_msg = f"Lỗi khởi tạo mô hình: {str(e)}"
            logger.error(error_msg)
            global _model_initialization_error
            _model_initialization_error = error_msg
            return jsonify({
                "success": False,
                "error": error_msg
            }), 500

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

        # Phát hiện các ô kệ
        detections, result_image = slot_model.detect_and_classify_slots(image)

        # Tính tóm tắt
        empty_count = sum(1 for d in detections if d["class"] == "empty_slot")
        filled_count = sum(1 for d in detections if d["class"] == "filled_slot")
        total_count = len(detections)

        # Mã hóa hình ảnh kết quả
        _, buffer = cv2.imencode('.jpg', result_image)
        result_image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Tính thời gian xử lý
        processing_time = time.time() - start_time

        # Trả về kết quả
        return jsonify({
            "success": True,
            "processing_time": processing_time,
            "detections": detections,
            "summary": {
                "total_slots": total_count,
                "empty_slots": empty_count,
                "filled_slots": filled_count,
                "occupancy_rate": filled_count / total_count if total_count > 0 else 0
            },
            "result_image": result_image_b64,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    except Exception as e:
        logger.error(f"Lỗi khi xử lý yêu cầu phát hiện ô kệ: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@slot_detection_bp.route('/slots_status', methods=['GET'])
def get_slots_status():
    """API endpoint trả về trạng thái các ô kệ dưới dạng tín hiệu đơn giản."""
    try:
        # Sử dụng biến toàn cục
        global _global_slot_model, _model_initialization_error

        # Lấy mô hình từ ứng dụng Flask hoặc biến toàn cục
        slot_model = current_app.config.get('slot_model') or _global_slot_model

        # Nếu mô hình chưa được khởi tạo, thử khởi tạo nó
        if not slot_model:
            # Nếu đã có lỗi khởi tạo trước đó, không thử lại
            if _model_initialization_error:
                return jsonify({
                    "success": False,
                    "error": f"Mô hình chưa được khởi tạo. Chi tiết: {_model_initialization_error}"
                }), 500

            try:
                model_path = current_app.config.get('model_path')
                device = current_app.config.get('device', 'cpu')
                logger.info(f"Đang khởi tạo mô hình phát hiện ô kệ từ {model_path}")
                slot_model = SlotDetectionModel(model_path=model_path, device=device)

                # Lưu mô hình vào cả config và biến toàn cục
                current_app.config['slot_model'] = slot_model
                _global_slot_model = slot_model
                logger.info("Đã khởi tạo mô hình phát hiện ô kệ thành công")
            except Exception as e:
                error_msg = f"Lỗi khởi tạo mô hình: {str(e)}"
                logger.error(error_msg)
                _model_initialization_error = error_msg
                return jsonify({
                    "success": False,
                    "error": f"Mô hình chưa được khởi tạo. Vui lòng khởi động lại server. Chi tiết: {error_msg}"
                }), 500

        # Nếu không có dữ liệu phát hiện trước đó, trả về mẫu dữ liệu trống
        if not hasattr(slot_model, 'last_detections') or not slot_model.last_detections:
            logger.warning("Chưa có dữ liệu phát hiện ô kệ. Trả về dữ liệu mẫu trống.")
            return jsonify({
                "success": True,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "signals": [],
                "note": "Chưa có dữ liệu phát hiện thực tế. Đây là dữ liệu mẫu."
            })

        # Tạo dữ liệu tín hiệu từ phát hiện trước đó
        signals = []
        for detection in slot_model.last_detections:
            slot_id = detection.get("slot_id", 0)
            has_goods = detection.get("class") == "filled_slot"
            row = detection.get("row", 0)
            column = detection.get("column", 0)

            signals.append({
                "id": slot_id,
                "position": f"{row}_{column}",
                "has_goods": has_goods,
                "confidence": detection.get("confidence", 0.0)
            })

        return jsonify({
            "success": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "signals": signals
        })

    except Exception as e:
        logger.error(f"Lỗi khi lấy trạng thái ô kệ: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@slot_detection_bp.route('/init_model', methods=['POST'])
def init_model():
    """API endpoint để khởi tạo mô hình khi cần."""
    try:
        global _global_slot_model, _model_initialization_error

        # Reset lỗi khởi tạo
        _model_initialization_error = None

        # Lấy đường dẫn và thiết bị từ config
        model_path = current_app.config.get('model_path')
        device = current_app.config.get('device', 'cpu')

        if not model_path:
            return jsonify({
                "success": False,
                "error": "Không tìm thấy đường dẫn mô hình trong cấu hình"
            }), 400

        logger.info(f"Đang khởi tạo mô hình phát hiện ô kệ từ {model_path}")
        slot_model = SlotDetectionModel(model_path=model_path, device=device)

        # Lưu mô hình vào cả config và biến toàn cục
        current_app.config['slot_model'] = slot_model
        _global_slot_model = slot_model

        return jsonify({
            "success": True,
            "message": "Đã khởi tạo mô hình phát hiện ô kệ thành công",
            "model_path": model_path,
            "device": device
        })

    except Exception as e:
        error_msg = f"Lỗi khi khởi tạo mô hình: {str(e)}"
        logger.error(error_msg)
        _model_initialization_error = error_msg
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500


def register_slot_detection_endpoints(app, model_path=None, device=None):
    """
    Đăng ký các endpoint phát hiện ô kệ với ứng dụng Flask.

    Args:
        app: Ứng dụng Flask
        model_path: Đường dẫn đến mô hình
        device: Thiết bị GPU hoặc CPU
    """
    app.config['model_path'] = model_path
    app.config['device'] = device

    # Đăng ký blueprint
    app.register_blueprint(slot_detection_bp, url_prefix='/api')

    # Khởi tạo mô hình ngay khi đăng ký endpoints
    try:
        global _global_slot_model
        logger.info(f"Đang khởi tạo mô hình phát hiện ô kệ từ {model_path}")
        slot_model = SlotDetectionModel(model_path=model_path, device=device)
        app.config['slot_model'] = slot_model
        _global_slot_model = slot_model
        logger.info("Đã khởi tạo mô hình phát hiện ô kệ thành công")
    except Exception as e:
        global _model_initialization_error
        error_msg = f"Lỗi khởi tạo mô hình khi đăng ký endpoints: {str(e)}"
        _model_initialization_error = error_msg
        logger.error(error_msg)
        # Vẫn đăng ký blueprint nhưng ghi nhận lỗi