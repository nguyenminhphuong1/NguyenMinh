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
import cv2

# Thiết lập logger
logger = logging.getLogger(__name__)


class SlotDetectionModel:
    """Mô hình phát hiện ô trống/có hàng trong kho bằng Fast R-CNN."""

    def __init__(self, model_path: str = None, device: str = None, confidence_threshold: float = 0.5):
        """
        Khởi tạo mô hình phát hiện ô kệ.

        Args:
            model_path: Đường dẫn đến tệp trọng số đã huấn luyện
            device: Thiết bị sử dụng ('cuda' hoặc 'cpu')
            confidence_threshold: Ngưỡng tin cậy cho phát hiện
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.num_classes = 3  # background, empty_slot, filled_slot

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

        logger.info(f"Khởi tạo mô hình phát hiện ô kệ trên {self.device}")

        # Khởi tạo mô hình
        self.model = self._create_model()

        # Chuyển mô hình sang thiết bị và chế độ đánh giá
        self.model.to(self.device)
        self.model.eval()

        # Danh sách các lớp (lớp 0 là nền)
        self.classes = ["background", "empty_slot", "filled_slot"]

        # Ánh xạ các chỉ số lớp thành nhãn lớp
        self.class_mapping = {i: cls for i, cls in enumerate(self.classes)}

        # Khởi tạo bộ phát hiện dựa trên ô cố định
        self._init_fixed_slots()

        # Lưu kết quả phát hiện gần nhất
        self.last_detections = []

    def _create_model(self) -> FasterRCNN:
        """Tạo và tải mô hình Fast R-CNN."""
        try:
            # Sử dụng mô hình đã được huấn luyện sẵn từ torchvision
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

    def _init_fixed_slots(self):
        """Khởi tạo các ô kệ cố định dựa trên cấu trúc kho hàng."""
        # Định nghĩa các ô kệ cố định dựa trên kiến trúc kho hàng
        # Format: [x1, y1, x2, y2, row, column]
        self.fixed_slots = [
            # Dãy kệ trên - 8 ô
            [50, 150, 150, 250, 0, 0],  # Ô 1
            [150, 150, 250, 250, 0, 1],  # Ô 2
            [250, 150, 350, 250, 0, 2],  # Ô 3
            [350, 150, 450, 250, 0, 3],  # Ô 4
            [450, 150, 550, 250, 0, 4],  # Ô 5
            [550, 150, 650, 250, 0, 5],  # Ô 6
            [650, 150, 750, 250, 0, 6],  # Ô 7
            [750, 150, 850, 250, 0, 7],  # Ô 8

            # Dãy kệ dưới - 9 ô
            [50, 350, 150, 450, 1, 0],  # Ô 9
            [150, 350, 250, 450, 1, 1],  # Ô 10
            [250, 350, 350, 450, 1, 2],  # Ô 11
            [350, 350, 450, 450, 1, 3],  # Ô 12
            [450, 350, 550, 450, 1, 4],  # Ô 13
            [550, 350, 650, 450, 1, 5],  # Ô 14
            [650, 350, 750, 450, 1, 6],  # Ô 15
            [750, 350, 850, 450, 1, 7],  # Ô 16
            [850, 350, 950, 450, 1, 8]  # Ô 17
        ]

    def _adjust_slots_to_image(self, image_height, image_width):
        """Điều chỉnh vị trí các ô kệ dựa trên kích thước hình ảnh."""
        adjusted_slots = []

        # Tính toán tỷ lệ hình ảnh
        for slot in self.fixed_slots:
            # Điều chỉnh tọa độ theo kích thước thực tế của hình ảnh
            x1 = int(slot[0] * image_width / 1000)
            y1 = int(slot[1] * image_height / 500)
            x2 = int(slot[2] * image_width / 1000)
            y2 = int(slot[3] * image_height / 500)
            row = slot[4]
            col = slot[5]

            adjusted_slots.append([x1, y1, x2, y2, row, col])

        return adjusted_slots

    def detect_fixed_slots(self, image: np.ndarray) -> List[Dict]:
        """
        Phát hiện trạng thái của các ô kệ cố định.

        Args:
            image: Hình ảnh numpy.ndarray (H, W, C)

        Returns:
            List[Dict]: Danh sách các ô kệ với thông tin vị trí và trạng thái
        """
        if image is None:
            logger.error("Hình ảnh đầu vào rỗng")
            return []

        image_height, image_width = image.shape[:2]
        adjusted_slots = self._adjust_slots_to_image(image_height, image_width)

        detections = []

        for i, (x1, y1, x2, y2, row, col) in enumerate(adjusted_slots):
            # Trích xuất vùng ảnh của ô kệ
            roi = image[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Phát hiện có hàng dựa trên phân tích màu sắc và đặc trưng
            # Đây là một thuật toán đơn giản, có thể cải thiện bằng ML
            filled = self._is_slot_filled(roi)

            # Xác định lớp và độ tin cậy
            class_name = "filled_slot" if filled else "empty_slot"
            confidence = 0.85 if filled else 0.75  # Giả định độ tin cậy

            detection = {
                "bbox": [x1, y1, x2, y2],
                "class": class_name,
                "confidence": confidence,
                "row": row,
                "column": col,
                "slot_id": i + 1
            }

            detections.append(detection)

        return detections

    def _is_slot_filled(self, roi: np.ndarray) -> bool:
        """
        Xác định ô kệ có hàng hay không dựa trên đặc trưng hình ảnh.

        Args:
            roi: Vùng ảnh của ô kệ

        Returns:
            bool: True nếu ô có hàng, False nếu ô trống
        """
        try:
            # Chuyển đổi sang thang xám
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # Tính histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

            # Áp dụng ngưỡng Otsu
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Tính tỷ lệ pixel trắng
            white_ratio = np.sum(thresh > 0) / thresh.size

            # Tính độ lệch chuẩn của độ sáng
            std_dev = np.std(gray)

            # Phát hiện cạnh
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size

            # Các tiêu chí phân loại (có thể điều chỉnh)
            if white_ratio > 0.5 and std_dev < 50:
                # Ô trống thường có màu đồng nhất
                return False
            elif edge_ratio > 0.1 or std_dev > 40:
                # Ô có hàng thường có nhiều cạnh và độ tương phản cao
                return True
            else:
                # Mặc định là trống nếu không chắc chắn
                return False

        except Exception as e:
            logger.error(f"Lỗi khi phân tích ô kệ: {str(e)}")
            return False

    def detect_and_classify_slots(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Phát hiện và phân loại các ô kệ trong hình ảnh.

        Args:
            image: Hình ảnh numpy.ndarray (H, W, C)

        Returns:
            Tuple[List[Dict], np.ndarray]: (danh sách phát hiện, hình ảnh đã vẽ kết quả)
        """
        if image is None:
            return [], None

        # Phát hiện các ô kệ cố định
        detections = self.detect_fixed_slots(image)

        # Lưu kết quả phát hiện gần nhất
        self.last_detections = detections

        # Vẽ kết quả lên hình ảnh
        result_image = self.draw_detection_results(image, detections)

        return detections, result_image

    def draw_detection_results(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Vẽ kết quả phát hiện lên hình ảnh.

        Args:
            image: Hình ảnh gốc
            detections: Danh sách các phát hiện

        Returns:
            np.ndarray: Hình ảnh đã vẽ kết quả
        """
        if image is None:
            return None

        result_image = image.copy()

        # Đếm số lượng ô trống và có hàng
        empty_count = sum(1 for d in detections if d["class"] == "empty_slot")
        filled_count = sum(1 for d in detections if d["class"] == "filled_slot")

        # Vẽ các ô kệ
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            slot_id = detection.get("slot_id", 0)

            # Chọn màu dựa trên lớp
            if class_name == "empty_slot":
                color = (0, 255, 0)  # Xanh lá cho ô trống
            else:
                color = (0, 0, 255)  # Đỏ cho ô có hàng

            # Vẽ hộp giới hạn
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # Vẽ nhãn
            label = f"#{slot_id}: {class_name.split('_')[0]}"
            cv2.putText(
                result_image,
                label,
                (x1 + 5, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Vẽ thông tin tổng quát
        status_text = f"Trang thai ke: {filled_count}/{len(detections)} o co hang ({empty_count} o trong)"
        cv2.putText(
            result_image,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),  # Màu trắng
            2
        )

        # Vẽ thời gian
        time_text = time.strftime("%d-%m-%Y %H:%M:%S")
        cv2.putText(
            result_image,
            time_text,
            (10, result_image.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # Màu trắng
            2
        )

        return result_image

    def predict(self, image: np.ndarray) -> Tuple[bool, List[Dict], float]:
        """
        Phát hiện xem có hàng hóa trong ảnh hay không.

        Args:
            image: Hình ảnh numpy.ndarray (H, W, C)

        Returns:
            Tuple[bool, List[Dict], float]: (có_hàng, danh_sách_phát_hiện, độ_tin_cậy_tổng_thể)
        """
        detections, _ = self.detect_and_classify_slots(image)

        # Xác định có hàng hay không
        has_goods = any(d["class"] == "filled_slot" for d in detections)

        # Tính độ tin cậy tổng thể
        if detections:
            filled_confidences = [d["confidence"] for d in detections if d["class"] == "filled_slot"]
            overall_confidence = max(filled_confidences) if filled_confidences else 0.0
        else:
            overall_confidence = 0.0

        return has_goods, detections, overall_confidence