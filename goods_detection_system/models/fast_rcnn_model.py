import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import logging
import time
import os
from typing import Dict, List, Tuple, Optional, Union

# Thiết lập logger
logger = logging.getLogger(__name__)


class FastRCNNModel:
    """Mô hình Fast R-CNN cho phát hiện đối tượng."""

    def __init__(self, model_path: str, num_classes: int = 6, device: str = None, confidence_threshold: float = 0.5):
        """
        Khởi tạo mô hình Fast R-CNN.

        Args:
            model_path: Đường dẫn đến tệp trọng số đã huấn luyện
            num_classes: Số lượng lớp (bao gồm lớp nền)
            device: Thiết bị sử dụng ('cuda' hoặc 'cpu')
            confidence_threshold: Ngưỡng tin cậy cho phát hiện
        """
        self.model_path = model_path
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold

        # Xác định thiết bị để chạy mô hình
        if device == 'cuda':
            # Kiểm tra xem có GPU không trước khi sử dụng CUDA
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                logger.warning("CUDA được yêu cầu nhưng không có GPU. Sử dụng CPU thay thế.")
                self.device = torch.device('cpu')
        else:
            self.device = torch.device('cpu')

        logger.info(f"Khởi tạo mô hình Fast R-CNN trên {self.device}")

        # Khởi tạo mô hình
        self.model = self._create_model()

        # Chuyển mô hình sang thiết bị và chế độ đánh giá
        self.model.to(self.device)
        self.model.eval()

        # Danh sách các lớp (lớp 0 là nền)
        self.classes = ["background", "box", "package", "container", "product", "pallet"]

        # Ánh xạ các chỉ số lớp thành nhãn lớp
        self.class_mapping = {i: cls for i, cls in enumerate(self.classes)}

    def _create_model(self) -> FasterRCNN:
        """Tạo và tải mô hình Fast R-CNN."""
        try:
            # Sử dụng mô hình đã được huấn luyện sẵn từ torchvision thay vì tự tạo
            logger.info("Tạo mô hình Faster R-CNN từ torchvision")
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

            # Điều chỉnh lớp đầu ra cho số lượng lớp cần thiết
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

            # Tải trọng số đã huấn luyện nếu có
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"Đang tải trọng số từ {self.model_path}")
                state_dict = torch.load(self.model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                logger.info("Đã tải trọng số thành công")
            else:
                logger.warning("Không tìm thấy file trọng số. Sử dụng mô hình đã huấn luyện sẵn.")

            return model

        except Exception as e:
            logger.error(f"Lỗi khi tạo mô hình Fast R-CNN: {str(e)}")
            # Tạo một mô hình đơn giản trong trường hợp lỗi
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            return model

    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Thực hiện dự đoán trên hình ảnh.

        Args:
            image: Hình ảnh numpy.ndarray (H, W, C) với định dạng BGR hoặc RGB

        Returns:
            List[Dict]: Danh sách các phát hiện với các thông tin như hộp giới hạn, lớp, độ tin cậy
        """
        if image is None:
            logger.error("Hình ảnh đầu vào rỗng")
            return []

        try:
            # Đảm bảo hình ảnh có định dạng RGB
            if image.shape[2] == 3:
                # Chuyển đổi từ BGR sang RGB nếu cần thiết
                rgb_image = image if image[0, 0, 0] <= image[0, 0, 2] else image[:, :, ::-1]
            else:
                logger.error(f"Hình ảnh có số kênh không hợp lệ: {image.shape[2]}")
                return []

            # Chuyển đổi sang tensor PyTorch
            tensor_image = torch.from_numpy(rgb_image.copy()).permute(2, 0, 1).float() / 255.0
            tensor_image = tensor_image.to(self.device)

            # Thêm chiều batch
            batch = [tensor_image]

            # Đo thời gian dự đoán
            start_time = time.time()

            # Thực hiện dự đoán
            with torch.no_grad():
                predictions = self.model(batch)

            # Tính thời gian dự đoán
            inference_time = time.time() - start_time
            logger.debug(f"Thời gian dự đoán: {inference_time:.4f}s")

            # Xử lý kết quả dự đoán
            results = []

            # Phân tích kết quả từ mô hình
            for i, pred in enumerate(predictions):
                boxes = pred['boxes'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()

                # Lọc các phát hiện dựa trên ngưỡng tin cậy
                valid_indices = np.where(scores >= self.confidence_threshold)[0]

                for idx in valid_indices:
                    # Lấy thông tin phát hiện
                    box = boxes[idx].astype(int)
                    score = float(scores[idx])
                    label_id = int(labels[idx])

                    # Lấy nhãn lớp
                    class_name = self.class_mapping.get(label_id, "unknown")

                    # Tạo đối tượng kết quả
                    detection = {
                        "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        "class": class_name,
                        "confidence": score
                    }

                    results.append(detection)

            return results

        except Exception as e:
            logger.error(f"Lỗi khi thực hiện dự đoán: {str(e)}")
            return []

    def detect_goods(self, image: np.ndarray) -> Tuple[bool, List[Dict], float]:
        """
        Phát hiện xem có hàng hóa trong hình ảnh hay không.

        Args:
            image: Hình ảnh numpy.ndarray (H, W, C)

        Returns:
            Tuple[bool, List[Dict], float]: (có_hàng, danh_sách_phát_hiện, độ_tin_cậy_tổng_thể)
        """
        # Thực hiện dự đoán
        detections = self.predict(image)

        # Nếu không có phát hiện, trả về False
        if not detections:
            return False, [], 0.0

        # Tính toán độ tin cậy tổng thể dựa trên các phát hiện
        confidences = [detection["confidence"] for detection in detections]
        overall_confidence = max(confidences) if confidences else 0.0

        # Xác định có hàng hóa hay không
        has_goods = overall_confidence >= self.confidence_threshold

        return has_goods, detections, overall_confidence