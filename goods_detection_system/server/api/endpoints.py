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
from models.fast_rcnn_model import FastRCNNModel
from utils.image_processor import ImageProcessor

# Thiết lập logger
logger = logging.getLogger(__name__)

# Tạo blueprint
api_bp = Blueprint('api', __name__)


@api_bp.route('/detect', methods=['POST'])
def detect():
    """API endpoint để phát hiện hàng hóa trong hình ảnh."""
    start_time = time.time()

    # Lấy mô hình và bộ xử lý hình ảnh từ ứng dụng Flask
    model = current_app.config['model']
    image_processor = current_app.config['image_processor']

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

        # Tiền xử lý hình ảnh
        processed_image = image_processor.preprocess(image)

        # Phát hiện hàng hóa
        has_goods, detections, confidence = model.detect_goods(processed_image)

        # Mã hóa hình ảnh kết quả với các bounding box
        result_image = None
        if data.get("return_image", False) and detections:
            # Vẽ kết quả phát hiện lên hình ảnh
            result_image = image_processor.draw_detection_results(
                image, detections, confidence_threshold=model.confidence_threshold
            )

            # Thêm dấu thời gian
            result_image = image_processor.add_timestamp(result_image)

            # Mã hóa hình ảnh kết quả
            _, buffer = cv2.imencode('.jpg', result_image)
            result_image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Cập nhật thống kê
        processing_time = time.time() - start_time

        # Trả về kết quả
        response = {
            "success": True,
            "has_goods": has_goods,
            "confidence": confidence,
            "detections": detections,
            "processing_time": processing_time,
            "camera_id": data.get("camera_id", "unknown"),
            "timestamp": data.get("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        }

        if result_image is not None:
            response["result_image"] = result_image_b64

        return jsonify(response)

    except Exception as e:
        logger.error(f"Lỗi khi xử lý yêu cầu phát hiện: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    """API endpoint kiểm tra trạng thái hoạt động của server."""
    model = current_app.config.get('model')

    if model:
        device = str(model.device)
        model_loaded = True
    else:
        device = "unknown"
        model_loaded = False

    return jsonify({
        "status": "ok",
        "version": "1.0.0",
        "uptime": time.time() - current_app.config.get('start_time', time.time()),
        "model_loaded": model_loaded,
        "device": device
    })


@api_bp.route('/cameras', methods=['GET'])
def get_cameras():
    """API endpoint để lấy danh sách camera."""
    camera_manager = current_app.config.get('camera_manager')

    if not camera_manager:
        return jsonify({
            "success": False,
            "error": "Không có camera manager"
        }), 500

    cameras = camera_manager.get_all_cameras_info()

    return jsonify({
        "success": True,
        "cameras": cameras
    })


def register_endpoints(app, model, image_processor, camera_manager=None):
    """
    Đăng ký các endpoint API với ứng dụng Flask.

    Args:
        app: Ứng dụng Flask
        model: Mô hình FastRCNN
        image_processor: Bộ xử lý hình ảnh
        camera_manager: Quản lý camera (tùy chọn)
    """
    app.config['model'] = model
    app.config['image_processor'] = image_processor
    app.config['camera_manager'] = camera_manager
    app.config['start_time'] = time.time()

    app.register_blueprint(api_bp, url_prefix='/api')


# Thêm vào file server/api/endpoints.py

@api_bp.route('/model/list', methods=['GET'])
def list_models():
    """Liệt kê tất cả các mô hình đã huấn luyện."""
    try:
        models_dir = os.path.join(ROOT_DIR, "models/weights")
        models = [f for f in os.listdir(models_dir) if f.endswith('.pth')]

        model_info = []
        for model in models:
            model_path = os.path.join(models_dir, model)
            model_info.append({
                "name": model,
                "size": os.path.getsize(model_path),
                "last_modified": os.path.getmtime(model_path)
            })

        return jsonify({
            "success": True,
            "models": model_info
        })
    except Exception as e:
        logger.error(f"Lỗi khi liệt kê mô hình: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@api_bp.route('/model/activate', methods=['POST'])
def activate_model():
    """Kích hoạt một mô hình đã huấn luyện."""
    try:
        data = request.json
        if not data or "model_name" not in data:
            return jsonify({
                "success": False,
                "error": "Thiếu tham số model_name"
            }), 400

        model_name = data["model_name"]
        model_path = os.path.join(ROOT_DIR, "models/weights", model_name)

        if not os.path.exists(model_path):
            return jsonify({
                "success": False,
                "error": f"Không tìm thấy mô hình: {model_name}"
            }), 404

        # Cập nhật cấu hình
        import config.settings as settings
        settings.AI_SERVER["model_path"] = model_path

        # Tải lại mô hình
        from models.fast_rcnn_model import FastRCNNModel
        device = current_app.config['model'].device
        new_model = FastRCNNModel(
            model_path=model_path,
            num_classes=6,
            device=device,
            confidence_threshold=settings.AI_SERVER.get("threshold", 0.7)
        )

        current_app.config['model'] = new_model

        return jsonify({
            "success": True,
            "message": f"Đã kích hoạt mô hình: {model_name}"
        })
    except Exception as e:
        logger.error(f"Lỗi khi kích hoạt mô hình: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@api_bp.route('/model/upload', methods=['POST'])
def upload_model():
    """Tải lên một mô hình đã huấn luyện."""
    try:
        if 'model' not in request.files:
            return jsonify({
                "success": False,
                "error": "Không tìm thấy file mô hình"
            }), 400

        model_file = request.files['model']
        if model_file.filename == '':
            return jsonify({
                "success": False,
                "error": "Không có file nào được chọn"
            }), 400

        if not model_file.filename.endswith('.pth'):
            return jsonify({
                "success": False,
                "error": "File không phải định dạng .pth"
            }), 400

        model_path = os.path.join(ROOT_DIR, "models/weights", model_file.filename)
        model_file.save(model_path)

        return jsonify({
            "success": True,
            "message": f"Đã tải lên mô hình: {model_file.filename}"
        })
    except Exception as e:
        logger.error(f"Lỗi khi tải lên mô hình: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500