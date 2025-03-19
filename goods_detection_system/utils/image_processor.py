import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Union

# Thiết lập logger
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Xử lý hình ảnh trước khi gửi đến mô hình AI."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.resize_dim = self.config.get("resize_dimensions", (640, 640))
        self.normalize = self.config.get("normalize", True)
        self.use_gpu = self.config.get("use_gpu", False)

        # Nếu sử dụng GPU, kiểm tra xem OpenCV có được biên dịch với CUDA không
        if self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.use_gpu = True
            logger.info("Đã phát hiện GPU. Sử dụng GPU để xử lý hình ảnh.")
        else:
            self.use_gpu = False
            logger.info("Không phát hiện GPU hoặc CUDA không được hỗ trợ. Sử dụng CPU để xử lý hình ảnh.")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Tiền xử lý khung hình cho mô hình Fast R-CNN."""
        if frame is None:
            logger.warning("Không thể xử lý khung hình rỗng.")
            return None

        try:
            # Chuyển đổi màu nếu cần thiết
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:  # RGBA
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Thay đổi kích thước khung hình
            if self.use_gpu:
                # Sử dụng GPU để thay đổi kích thước
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                gpu_resized = cv2.cuda.resize(gpu_frame, self.resize_dim)
                processed_frame = gpu_resized.download()
            else:
                # Sử dụng CPU để thay đổi kích thước
                processed_frame = cv2.resize(frame, self.resize_dim, interpolation=cv2.INTER_AREA)

            # Chuẩn hóa pixel nếu được yêu cầu
            if self.normalize:
                processed_frame = processed_frame.astype(np.float32) / 255.0

            return processed_frame

        except Exception as e:
            logger.error(f"Lỗi khi tiền xử lý khung hình: {str(e)}")
            return None

    def draw_detection_results(self, frame: np.ndarray, detections: List[Dict],
                               confidence_threshold: float = 0.5) -> np.ndarray:
        """Vẽ kết quả phát hiện lên khung hình gốc."""
        if frame is None or detections is None:
            return frame

        result_frame = frame.copy()
        height, width = result_frame.shape[:2]

        # Màu cho các lớp khác nhau (BGR format)
        colors = {
            "box": (0, 255, 0),  # Green
            "package": (255, 0, 0),  # Blue
            "container": (0, 0, 255),  # Red
            "product": (255, 255, 0),  # Cyan
            "pallet": (255, 0, 255),  # Magenta
            "person": (0, 255, 255),  # Yellow
            "default": (128, 128, 128)  # Gray
        }

        for detection in detections:
            # Lọc các phát hiện có độ tin cậy thấp
            confidence = detection.get("confidence", 0)
            if confidence < confidence_threshold:
                continue

            # Lấy thông tin phát hiện
            class_name = detection.get("class", "default")
            box = detection.get("bbox")

            if box is None:
                continue

            # Chuyển đổi giá trị hộp giới hạn thành pixel
            # Định dạng box có thể là [x, y, width, height] hoặc [x1, y1, x2, y2]
            if len(box) == 4:
                if box[2] <= 1.0 and box[3] <= 1.0:  # Normalized format [x, y, w, h]
                    x1 = int(box[0] * width)
                    y1 = int(box[1] * height)
                    x2 = int((box[0] + box[2]) * width)
                    y2 = int((box[1] + box[3]) * height)
                elif box[2] > box[0] and box[3] > box[1]:  # Format [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, box)
                else:  # Format [x, y, w, h]
                    x1, y1 = int(box[0]), int(box[1])
                    x2, y2 = int(box[0] + box[2]), int(box[1] + box[3])
            else:
                logger.warning(f"Định dạng hộp giới hạn không được hỗ trợ: {box}")
                continue

            # Giới hạn tọa độ trong phạm vi khung hình
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)

            # Lấy màu cho lớp
            color = colors.get(class_name, colors["default"])

            # Vẽ hộp giới hạn
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)

            # Vẽ nhãn
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            # Vẽ nền cho nhãn
            cv2.rectangle(
                result_frame,
                (x1, y1 - label_height - 5),
                (x1 + label_width, y1),
                color,
                -1
            )

            # Vẽ văn bản nhãn
            cv2.putText(
                result_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        return result_frame

    @staticmethod
    def crop_region_of_interest(frame: np.ndarray, roi: List[Tuple[int, int]]) -> np.ndarray:
        """Cắt vùng quan tâm từ khung hình dựa trên tọa độ đa giác."""
        if frame is None or roi is None:
            return frame

        # Tạo mặt nạ từ tọa độ ROI
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        roi_points = np.array(roi, dtype=np.int32)
        cv2.fillPoly(mask, [roi_points], 255)

        # Áp dụng mặt nạ
        roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Tìm hộp giới hạn của ROI
        x, y, w, h = cv2.boundingRect(roi_points)

        # Cắt hộp giới hạn
        cropped = roi_frame[y:y + h, x:x + w]

        return cropped

    @staticmethod
    def enhance_contrast(frame: np.ndarray, clip_limit: float = 2.0,
                         tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """Cải thiện độ tương phản của hình ảnh bằng CLAHE."""
        if frame is None:
            return None

        try:
            # Chuyển đổi sang không gian màu LAB
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

            # Tách các kênh
            l, a, b = cv2.split(lab)

            # Áp dụng CLAHE cho kênh L
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            cl = clahe.apply(l)

            # Hợp nhất các kênh
            enhanced_lab = cv2.merge((cl, a, b))

            # Chuyển đổi trở lại không gian màu BGR
            enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            return enhanced_bgr

        except Exception as e:
            logger.error(f"Lỗi khi cải thiện độ tương phản: {str(e)}")
            return frame

    @staticmethod
    def reduce_noise(frame: np.ndarray, strength: int = 7) -> np.ndarray:
        """Giảm nhiễu trong hình ảnh bằng bộ lọc song phương."""
        if frame is None:
            return None

        try:
            # Áp dụng bộ lọc song phương để giảm nhiễu nhưng vẫn giữ cạnh
            denoised = cv2.bilateralFilter(frame, d=strength, sigmaColor=75, sigmaSpace=75)
            return denoised

        except Exception as e:
            logger.error(f"Lỗi khi giảm nhiễu: {str(e)}")
            return frame

    def add_timestamp(self, frame: np.ndarray, text: str = None) -> np.ndarray:
        """Thêm dấu thời gian vào khung hình."""
        if frame is None:
            return None

        timestamp = text or time.strftime("%Y-%m-%d %H:%M:%S")
        height, width = frame.shape[:2]

        # Vẽ nền cho timestamp
        cv2.rectangle(
            frame,
            (10, height - 40),
            (230, height - 10),
            (0, 0, 0),
            -1
        )

        # Vẽ timestamp
        cv2.putText(
            frame,
            timestamp,
            (15, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return frame