# Hệ thống phát hiện hàng hóa qua camera sử dụng Fast R-CNN

Dự án này xây dựng một hệ thống phát hiện hàng hóa qua camera IP và camera công nghiệp. Hệ thống sử dụng mô hình Fast R-CNN để nhận diện, phân tích hàng hóa và gửi tín hiệu tới các hệ thống khác. Hệ thống được thiết kế để chạy trên cả Linux và Windows.

## Tính năng chính

- Kết nối với camera IP và camera công nghiệp qua mạng không dây
- Phát hiện hàng hóa sử dụng mô hình Fast R-CNN
- Xử lý hình ảnh theo thời gian thực
- Giao diện người dùng trực quan
- Gửi tín hiệu có hàng/không có hàng tới server khác
- Hỗ trợ đa nền tảng (Linux và Windows)
- Xử lý đồng thời nhiều luồng camera
- Huấn luyện mô hình từ dữ liệu riêng (tùy chọn)

## Cài đặt

### Yêu cầu hệ thống

- Python 3.7+
- PyTorch 1.9.0+
- OpenCV 4.5.0+
- Flask 2.0.0+
- PyQt5 5.15.0+
- CUDA (tùy chọn, để tăng tốc độ xử lý nếu có GPU)

### Cài đặt thủ công

1. **Tạo môi trường ảo (khuyến nghị)**
   ```bash
   # Tạo môi trường ảo
   python -m venv venv

   # Kích hoạt môi trường
   # Linux/macOS
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```

2. **Cài đặt các thư viện cần thiết**
   ```bash
   pip install -r requirements.txt
   ```

3. **Tạo các thư mục cần thiết**
   ```bash
   mkdir -p models/weights
   mkdir -p data/goods_detection/images
   mkdir -p data/goods_detection/annotations
   mkdir -p logs
   ```

### Cài đặt tự động

Sử dụng script `setup.sh` (Linux/macOS) hoặc `setup.bat` (Windows):
```bash
# Linux/macOS
chmod +x setup.sh
./setup.sh

# Windows
setup.bat
```

## Cấu hình

### Cấu hình camera

Chỉnh sửa tệp `config/camera_config.json` để thiết lập thông tin kết nối camera:

```json
{
    "cameras": [
        {
            "id": "ip_camera_1",
            "name": "Camera IP 1",
            "type": "ip",
            "url": "rtsp://admin:password@192.168.1.100:554/stream1",
            "enabled": true
        },
        {
            "id": "webcam",
            "name": "Webcam cục bộ",
            "type": "webcam",
            "device_id": 0,
            "enabled": true
        }
    ]
}
```

### Cấu hình server AI

Chỉnh sửa tệp `config/settings.py` để thiết lập cấu hình server AI và thông báo:

```python
AI_SERVER = {
    "host": "localhost",
    "port": 5000,
    "model_path": "models/weights/fast_rcnn_model.pth",
    "threshold": 0.5,
    "device": "cpu"  # hoặc "cuda" nếu có GPU
}

NOTIFICATION_SERVER = {
    "url": "http://your-server-url/notification",
    "api_key": "your_api_key_here"
}
```

## Sử dụng cơ bản

### Chạy toàn bộ hệ thống (GUI và Server AI)

```bash
python main.py
```

### Chỉ chạy Server AI

```bash
python main.py --server
```

### Chỉ chạy GUI

```bash
python main.py --gui
```

### Chạy ở chế độ debug

```bash
python main.py --debug
```

## Hướng dẫn sử dụng

### Sử dụng giao diện đồ họa

1. **Kết nối camera**:
   - Nhấn nút "Kết nối tất cả camera" ở góc trên bên trái
   - Hệ thống sẽ cố gắng kết nối với các camera đã cấu hình

2. **Xem luồng video**:
   - Chọn camera từ dropdown (nếu có nhiều camera)
   - Luồng video sẽ hiển thị ở khung bên trái

3. **Cấu hình phát hiện**:
   - Đảm bảo checkbox "Bật phát hiện" đã được chọn
   - Điều chỉnh tần suất phát hiện nếu cần (mặc định là 5 khung hình)

4. **Xem kết quả phát hiện**:
   - Kết quả phát hiện sẽ hiển thị ở bảng điều khiển bên phải
   - Thông tin bao gồm trạng thái phát hiện, độ tin cậy và chi tiết về đối tượng đã phát hiện

### Sử dụng REST API

#### Endpoint phát hiện hàng hóa

- **URL**: `/detect`
- **Phương thức**: POST
- **Body**:
  ```json
  {
      "image": "base64_encoded_image_data",
      "camera_id": "camera_id_optional",
      "return_image": true
  }
  ```
- **Phản hồi thành công**:
  ```json
  {
      "success": true,
      "has_goods": true,
      "confidence": 0.95,
      "detections": [
          {
              "bbox": [100, 150, 300, 400],
              "class": "box",
              "confidence": 0.95
          }
      ],
      "processing_time": 0.125,
      "result_image": "base64_encoded_image_with_bounding_boxes"
  }
  ```

#### Endpoint kiểm tra trạng thái

- **URL**: `/health`
- **Phương thức**: GET
- **Phản hồi thành công**:
  ```json
  {
      "status": "ok",
      "version": "1.0.0",
      "uptime": 3600,
      "stats": {
          "total_requests": 50,
          "successful_detections": 45,
          "failed_detections": 5,
          "avg_processing_time": 0.1
      },
      "device": "cpu"
  }
  ```

#### Endpoint quản lý mô hình (nếu triển khai)

- **Liệt kê mô hình**: `GET /api/model/list`
- **Kích hoạt mô hình**: `POST /api/model/activate`
- **Tải lên mô hình**: `POST /api/model/upload`

#### Endpoint quản lý dữ liệu huấn luyện (nếu triển khai)

- **Thêm mẫu dữ liệu**: `POST /api/data/add_sample`
- **Xem thống kê**: `GET /api/data/stats`
- **Huấn luyện mô hình**: `POST /api/train`

### Ví dụ mã Python để sử dụng API

```python
import requests
import base64

# Đọc ảnh và mã hóa base64
with open("example.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Gửi yêu cầu phát hiện
response = requests.post(
    "http://localhost:5000/detect",
    json={
        "image": encoded_string,
        "return_image": True
    }
)

# Xử lý kết quả
if response.status_code == 200:
    result = response.json()
    print("Has goods:", result.get("has_goods"))
    print("Confidence:", result.get("confidence"))
    
    # Lưu ảnh kết quả
    if "result_image" in result:
        with open("result.jpg", "wb") as f:
            f.write(base64.b64decode(result["result_image"]))
        print("Result image saved to 'result.jpg'")
```

## Triển khai trên máy tính khác

### Phương pháp 1: Sao chép và cài đặt

1. Sao chép toàn bộ mã nguồn đến máy tính mới
2. Thực hiện quy trình cài đặt như hướng dẫn ở phần "Cài đặt"
3. Cấu hình lại các thông số trong `config/settings.py` và `config/camera_config.json`
4. Khởi động ứng dụng

### Phương pháp 2: Sử dụng Docker

1. **Đóng gói ứng dụng với Dockerfile**

   Tạo file `Dockerfile` với nội dung:
   ```dockerfile
   FROM python:3.8-slim

   WORKDIR /app

   # Cài đặt các thư viện hệ thống
   RUN apt-get update && apt-get install -y --no-install-recommends \
       build-essential \
       libopencv-dev \
       && rm -rf /var/lib/apt/lists/*

   # Sao chép requirements trước để tận dụng cache
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Sao chép mã nguồn
   COPY . .

   # Tạo các thư mục cần thiết
   RUN mkdir -p models/weights \
       data/goods_detection/images \
       data/goods_detection/annotations \
       logs

   # Thiết lập biến môi trường
   ENV PYTHONPATH=/app

   # Mở cổng cho server
   EXPOSE 5000

   # Lệnh khởi động
   CMD ["python", "main.py", "--server"]
   ```

2. **Tạo file Docker Compose**

   Tạo file `docker-compose.yml`:
   ```yaml
   version: '3'

   services:
     ai_server:
       build: .
       ports:
         - "5000:5000"
       volumes:
         - ./data:/app/data
         - ./models/weights:/app/models/weights
         - ./logs:/app/logs
       restart: unless-stopped
       environment:
         - DEBUG=0
         - USE_GPU=0
   ```

3. **Triển khai với Docker Compose**

   ```bash
   docker-compose up -d
   ```

### Phương pháp 3: Phân phối dưới dạng gói Python

1. Tạo file `setup.py`:
   ```python
   from setuptools import setup, find_packages

   setup(
       name="goods_detection",
       version="1.0.0",
       packages=find_packages(),
       install_requires=[
           "torch>=1.9.0",
           "torchvision>=0.10.0",
           "opencv-python>=4.5.0",
           "flask>=2.0.0",
           "pyqt5>=5.15.0",
           "numpy>=1.19.0",
           "pillow>=8.0.0",
           "requests>=2.25.0"
       ],
       entry_points={
           'console_scripts': [
               'goods-detection=goods_detection.main:main',
           ],
       },
   )
   ```

2. Cài đặt gói:
   ```bash
   pip install .
   ```

## Huấn luyện mô hình riêng

### Thu thập dữ liệu

1. Thu thập ảnh hàng hóa từ góc nhìn, ánh sáng, và khoảng cách khác nhau
2. Đặt ảnh vào thư mục `data/goods_detection/images`

### Gán nhãn dữ liệu

1. Sử dụng công cụ gán nhãn như [LabelImg](https://github.com/tzutalin/labelImg)
2. Gán nhãn và lưu file XML vào thư mục `data/goods_detection/annotations`

### Huấn luyện mô hình

Sử dụng API huấn luyện:
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "num_epochs": 10,
    "batch_size": 2,
    "learning_rate": 0.005
  }'
```

## Khắc phục sự cố

### Vấn đề về camera
- **Không kết nối được camera**: Kiểm tra URL, tên người dùng, mật khẩu
- **Luồng video chậm**: Điều chỉnh độ phân giải và FPS trong cấu hình

### Vấn đề về mô hình
- **Lỗi tải mô hình**: Kiểm tra đường dẫn đến file trọng số
- **Phát hiện không chính xác**: Điều chỉnh ngưỡng tin cậy hoặc huấn luyện lại mô hình

### Vấn đề về GPU
- **Lỗi CUDA**: Đảm bảo CUDA được cài đặt đúng cách hoặc chuyển sang sử dụng CPU
- **Thiếu bộ nhớ GPU**: Giảm kích thước batch hoặc cỡ hình trong quá trình huấn luyện

## Đóng góp

Các đóng góp là rất hoan nghênh! Vui lòng tạo issue hoặc pull request cho bất kỳ cải tiến nào.

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem tệp `LICENSE` để biết thêm chi tiết.
# Hướng dẫn sử dụng chức năng phát hiện ô kệ trong kho hàng

Chức năng phát hiện ô kệ là một tính năng mới được thêm vào hệ thống phát hiện hàng hóa Fast R-CNN. Tính năng này cho phép phân tích ảnh kho hàng, xác định các ô kệ và trạng thái của chúng (trống hoặc có hàng).

## Tính năng chính

- Phát hiện tự động vị trí các ô kệ trong ảnh kho hàng
- Phân loại ô kệ trống và ô kệ có hàng
- Tính toán tỷ lệ lấp đầy của kho hàng
- Hiển thị kết quả trực quan trên ảnh với các đánh dấu màu
- Tạo báo cáo thống kê về tình trạng kho hàng

## Cách sử dụng

### 1. Thông qua API

Để sử dụng chức năng này thông qua API, bạn có thể gửi yêu cầu POST đến endpoint `/detect` với tham số `detect_slots=true`:

```python
import requests
import base64

# Đọc ảnh và mã hóa base64
with open("warehouse_image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Gửi yêu cầu phát hiện
response = requests.post(
    "http://localhost:5000/detect",
    json={
        "image": encoded_string,
        "detect_slots": True,
        "return_image": True
    }
)

# Xử lý kết quả
if response.status_code == 200:
    result = response.json()
    
    # Hiển thị thông tin tổng kết
    summary = result.get("summary", {})
    print(f"Tổng số ô kệ: {summary.get('total_slots', 0)}")
    print(f"Số ô trống: {summary.get('empty_slots', 0)}")
    print(f"Số ô có hàng: {summary.get('filled_slots', 0)}")
    print(f"Tỷ lệ lấp đầy: {summary.get('occupancy_rate', 0):.2%}")
    
    # Lưu ảnh kết quả
    if "result_image" in result:
        with open("result.jpg", "wb") as f:
            f.write(base64.b64decode(result["result_image"]))
```

### 2. Sử dụng script kiểm thử

Sử dụng script `test_slot_detection.py` để kiểm thử chức năng:

```bash
python test_slot_detection.py path/to/warehouse_image.jpg
```

Script sẽ:
1. Kết nối với server AI
2. Gửi ảnh để phát hiện ô kệ
3. Hiển thị kết quả chi tiết về từng ô kệ và tình trạng kho hàng
4. Lưu ảnh kết quả với các đánh dấu trực quan

### 3. Thông qua giao diện người dùng

Nếu hệ thống có giao diện đồ họa:

1. Kết nối với camera quan sát kho hàng
2. Chọn chế độ "Phát hiện ô kệ" trong giao diện
3. Kết quả phát hiện sẽ hiển thị trực tiếp trên giao diện

## Cách đọc kết quả

Kết quả phát hiện ô kệ bao gồm:

1. **Tổng kết kho hàng**:
   - Tổng số ô kệ được phát hiện
   - Số ô trống
   - Số ô có hàng
   - Tỷ lệ lấp đầy

2. **Thông tin chi tiết từng ô**:
   - ID của ô
   - Vị trí (dãy, cột)
   - Trạng thái (trống/có hàng)
   - Độ tin cậy của phát hiện

3. **Hình ảnh kết quả**:
   - Ô trống: Khung màu xanh lá
   - Ô có hàng: Khung màu đỏ
   - Thông tin tổng kết trên ảnh

## Điều chỉnh và cấu hình

Để điều chỉnh vị trí các ô kệ cố định, bạn có thể chỉnh sửa mảng `fixed_slots` trong file `models/slot_detection_model.py`. Mỗi mục trong mảng có định dạng `[x1, y1, x2, y2, row, column]`:

- `x1, y1`: Tọa độ góc trên bên trái của ô
- `x2, y2`: Tọa độ góc dưới bên phải của ô
- `row`: Chỉ số dãy (0, 1, 2, ...)
- `column`: Chỉ số cột (0, 1, 2, ...)

## Giám sát thời gian thực

Để giám sát kho hàng theo thời gian thực:

1. Kết nối hệ thống với camera giám sát kho hàng
2. Cấu hình tần suất phát hiện (ví dụ: mỗi 5 giây)
3. Thiết lập thông báo khi phát hiện thay đổi trạng thái ô kệ
4. Lưu trữ dữ liệu lịch sử để phân tích xu hướng

## Cải thiện độ chính xác

Để cải thiện độ chính xác của phát hiện:

1. **Huấn luyện mô hình riêng**: Thu thập ảnh kho hàng cụ thể và gán nhãn các ô kệ
2. **Điều chỉnh thuật toán phân tích**: Chỉnh sửa hàm `_is_slot_filled()` để phù hợp với đặc điểm của hàng hóa
3. **Tối ưu ánh sáng**: Đảm bảo kho hàng có đủ ánh sáng và không có phản xạ gây nhiễu

---

Với chức năng phát hiện ô kệ, bạn có thể tự động hóa việc kiểm kê kho hàng, giám sát tình trạng sử dụng không gian và tối ưu hóa quản lý kho.
# Hướng dẫn sử dụng chức năng thông báo trạng thái ô kệ

Chức năng thông báo trạng thái ô kệ cho phép hệ thống gửi thông tin về tình trạng các ô kệ trong kho hàng đến một server khác để xử lý, giám sát hoặc hiển thị. Tài liệu này sẽ hướng dẫn các bước thiết lập và sử dụng chức năng này.

## 1. Tổng quan

Hệ thống phát hiện hàng hóa có thể:
- Phát hiện các ô kệ trong hình ảnh kho hàng
- Phân loại ô kệ trống và ô kệ có hàng
- Gửi thông báo về vị trí và trạng thái của từng ô kệ đến một server khác

## 2. Định dạng dữ liệu thông báo

Khi gửi thông báo, hệ thống sẽ tạo một đối tượng JSON với cấu trúc như sau:

```json
{
  "type": "slots_status",
  "timestamp": "2025-03-19 10:15:30",
  "camera_id": "warehouse_cam_1",
  "slots_data": [
    {
      "slot_id": 1,
      "has_goods": true,
      "position": {"row": 0, "column": 0},
      "confidence": 0.95
    },
    {
      "slot_id": 2,
      "has_goods": false,
      "position": {"row": 0, "column": 1},
      "confidence": 0.88
    },
    // ... các ô kệ khác
  ]
}
```

- `type`: Loại thông báo ("slots_status")
- `timestamp`: Thời gian phát hiện
- `camera_id`: ID của camera đang quan sát kho hàng
- `slots_data`: Mảng chứa thông tin về từng ô kệ
  - `slot_id`: ID của ô kệ
  - `has_goods`: `true` nếu ô có hàng, `false` nếu ô trống
  - `position`: Vị trí của ô kệ (dãy, cột)
  - `confidence`: Độ tin cậy của phát hiện (0.0 - 1.0)

## 3. Cấu hình server nhận thông báo

Để thiết lập server nhận thông báo, bạn cần cập nhật cấu hình trong file `config/settings.py`:

```python
# Cấu hình thông báo server
NOTIFICATION_SERVER = {
    "url": "http://your-server-url/api/endpoint",  # URL của server nhận thông báo
    "api_key": "your_api_key_here",  # API key để xác thực với server
    "retry_attempts": 3,  # Số lần thử lại nếu gửi thất bại
    "timeout": 5,  # Thời gian chờ (giây)
    "slots_notification": {
        "enabled": True,  # Bật/tắt tính năng thông báo ô kệ
        "min_interval": 1.0,  # Khoảng thời gian tối thiểu giữa các thông báo (giây)
        "send_on_change_only": True  # Chỉ gửi thông báo khi có sự thay đổi
    }
}
```

## 4. Kích hoạt chức năng thông báo

### Thông qua API

Khi gửi yêu cầu phát hiện ô kệ qua API, chỉ cần thêm tham số `detect_slots=true`:

```python
import requests
import base64

# Đọc ảnh và mã hóa base64
with open("warehouse_image.jpg", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# Gửi yêu cầu phát hiện với detect_slots=true
response = requests.post(
    "http://localhost:5000/detect",
    json={
        "image": encoded_string,
        "detect_slots": True,
        "camera_id": "warehouse_cam_1",  # Quan trọng để xác định nguồn
        "return_image": True
    }
)
```

Khi `detect_slots=true`, hệ thống sẽ tự động:
1. Phát hiện các ô kệ trong hình ảnh
2. Phân loại ô trống và ô có hàng
3. Gửi thông báo đến server được cấu hình

### Gửi thông báo thủ công

Bạn cũng có thể gửi thông báo thủ công bằng API `send_notification`:

```python
import requests

# Dữ liệu trạng thái ô kệ
slots_data = {
    "timestamp": "2025-03-19 10:15:30",
    "camera_id": "warehouse_cam_1",
    "slots_status": [
        {
            "slot_id": 1,
            "has_goods": True,
            "position": {"row": 0, "column": 0},
            "confidence": 0.95
        },
        # ... thêm các ô kệ khác
    ]
}

# Gửi thông báo
response = requests.post(
    "http://localhost:5000/api/send_notification",
    json={
        "type": "slots_status",
        "data": slots_data
    }
)
```

## 5. Kiểm thử chức năng thông báo

Sử dụng script `test_notification.py` để kiểm thử chức năng thông báo:

```bash
python test_notification.py
```

Script này sẽ:
1. Tạo dữ liệu giả lập về trạng thái các ô kệ
2. Gửi thông báo đến server
3. Hiển thị kết quả phản hồi

## 6. Tích hợp với hệ thống bên ngoài

### Server nhận thông báo

Server nhận thông báo cần có một endpoint API để nhận và xử lý dữ liệu. Ví dụ với Node.js:

```javascript
const express = require('express');
const app = express();
app.use(express.json());

app.post('/notification', (req, res) => {
  const notification = req.body;
  
  // Kiểm tra loại thông báo
  if (notification.type === 'slots_status') {
    // Xử lý thông báo trạng thái ô kệ
    console.log(`Nhận thông báo từ camera ${notification.camera_id}`);
    console.log(`Số ô kệ: ${notification.slots_data.length}`);
    
    // Đếm số ô trống và có hàng
    const filledSlots = notification.slots_data.filter(slot => slot.has_goods);
    console.log(`Ô có hàng: ${filledSlots.length}`);
    console.log(`Ô trống: ${notification.slots_data.length - filledSlots.length}`);
    
    // Lưu vào cơ sở dữ liệu, hiển thị trên dashboard, v.v.
    // ...
  }
  
  res.json({ success: true, message: 'Đã nhận thông báo' });
});

app.listen(8000, () => {
  console.log('Server đang lắng nghe tại http://localhost:8000');
});
```

### Xác thực API

Để đảm bảo an toàn, server nhận thông báo nên kiểm tra API key từ header:

```javascript
// Middleware kiểm tra API key
function verifyApiKey(req, res, next) {
  const apiKey = req.headers.authorization;
  
  if (!apiKey || !apiKey.startsWith('Bearer ')) {
    return res.status(401).json({ success: false, error: 'Không có API key' });
  }
  
  const key = apiKey.split(' ')[1];
  if (key !== 'your_api_key_here') {
    return res.status(403).json({ success: false, error: 'API key không hợp lệ' });
  }
  
  next();
}

// Áp dụng middleware
app.post('/notification', verifyApiKey, (req, res) => {
  // Xử lý thông báo
  // ...
});
```

## 7. Khắc phục sự cố

### Thông báo không được gửi đi

- Kiểm tra URL server trong cấu hình
- Kiểm tra kết nối mạng giữa hai server
- Đảm bảo API key chính xác
- Kiểm tra log trong thư mục `logs` của hệ thống

### Thông báo không chứa dữ liệu ô kệ

- Đảm bảo chức năng phát hiện ô kệ hoạt động chính xác
- Kiểm tra tham số `detect_slots=true` khi gửi yêu cầu
- Kiểm tra `NOTIFICATION_SERVER.slots_notification.enabled` đã được đặt thành `True`

### Thông báo quá nhiều hoặc quá ít

Điều chỉnh cấu hình trong `NOTIFICATION_SERVER.slots_notification`:
- `min_interval`: Tăng giá trị này để giảm tần suất thông báo
- `send_on_change_only`: Đặt thành `True` để chỉ gửi thông báo khi có sự thay đổi

---

Với hướng dẫn này, bạn có thể thiết lập và sử dụng chức năng thông báo trạng thái ô kệ trong hệ thống phát hiện hàng hóa để tích hợp với các hệ thống bên ngoài như phần mềm quản lý kho, hệ thống giám sát, hoặc dashboard hiển thị trạng thái kho hàng theo thời gian thực.
# Hướng dẫn sử dụng API tín hiệu ô kệ

Tài liệu này mô tả cách sử dụng API tín hiệu ô kệ để nhận thông tin về trạng thái các ô kệ (có hàng/không có hàng) thông qua một API đơn giản.

## 1. Tổng quan

API tín hiệu ô kệ cung cấp một endpoint đơn giản để truy vấn trạng thái hiện tại của tất cả các ô kệ trong kho hàng. API này trả về thông tin dưới dạng JSON với định dạng đơn giản và dễ tích hợp vào các hệ thống khác.

## 2. API Endpoint

### Lấy trạng thái ô kệ

- **URL**: `/api/slots_status`
- **Phương thức**: GET
- **Mô tả**: Trả về trạng thái hiện tại của tất cả các ô kệ

**Phản hồi thành công (200 OK)**:
```json
{
  "success": true,
  "timestamp": "2025-03-19 10:15:30",
  "signals": [
    {
      "id": 1,
      "position": "0_0",
      "has_goods": true,
      "confidence": 0.95
    },
    {
      "id": 2,
      "position": "0_1",
      "has_goods": false,
      "confidence": 0.88
    },
    // ... các ô kệ khác
  ]
}
```

**Phản hồi lỗi (404 Not Found)**:
```json
{
  "success": false,
  "error": "Chưa có dữ liệu phát hiện ô kệ. Hãy thực hiện phát hiện trước."
}
```

## 3. Các trường dữ liệu

- `success`: `true` nếu yêu cầu thành công, `false` nếu có lỗi
- `timestamp`: Thời gian cập nhật trạng thái
- `signals`: Mảng chứa thông tin từng ô kệ:
  - `id`: ID của ô kệ (số nguyên)
  - `position`: Vị trí của ô kệ theo định dạng "dãy_cột"
  - `has_goods`: `true` nếu ô có hàng, `false` nếu ô trống
  - `confidence`: Độ tin cậy của phát hiện (0.0 - 1.0)

## 4. Cập nhật trạng thái ô kệ

Trước khi có thể lấy trạng thái ô kệ, bạn cần cập nhật dữ liệu bằng cách thực hiện phát hiện ô kệ. Bạn có thể làm điều này bằng cách gửi yêu cầu phát hiện đến endpoint `/detect`:

```bash
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image_data",
    "detect_slots": true,
    "camera_id": "warehouse_cam_1"
  }'
```

Hoặc bằng cách sử dụng script test đã cung cấp:

```bash
python test_slot_detection.py ./images/warehouse_image.jpg
```

## 5. Sử dụng API trong các hệ thống khác

### Python

```python
import requests
import time

def get_slot_signals(server_url="http://localhost:5000"):
    response = requests.get(f"{server_url}/api/slots_status")
    
    if response.status_code == 200:
        data = response.json()
        return data.get("signals", [])
    else:
        print(f"Lỗi: {response.status_code}, {response.text}")
        return []

# Lấy trạng thái ô kệ
signals = get_slot_signals()

# Phân tích trạng thái
filled = [s for s in signals if s.get("has_goods")]
empty = [s for s in signals if not s.get("has_goods")]

print(f"Tổng số ô: {len(signals)}")
print(f"Số ô có hàng: {len(filled)}")
print(f"Số ô trống: {len(empty)}")
```

### JavaScript

```javascript
fetch('http://localhost:5000/api/slots_status')
  .then(response => response.json())
  .then(data => {
    if (data.success) {
      const signals = data.signals;
      
      // Đếm các ô trống và có hàng
      const filled = signals.filter(s => s.has_goods);
      const empty = signals.filter(s => !s.has_goods);
      
      console.log(`Tổng số ô: ${signals.length}`);
      console.log(`Số ô có hàng: ${filled.length}`);
      console.log(`Số ô trống: ${empty.length}`);
      
      // Hiển thị thông tin từng ô
      signals.forEach(signal => {
        console.log(`Ô #${signal.id}, Vị trí: ${signal.position}, ${signal.has_goods ? 'CÓ HÀNG' : 'TRỐNG'}`);
      });
    } else {
      console.error(data.error);
    }
  })
  .catch(error => console.error('Lỗi:', error));
```

## 6. Giám sát theo thời gian thực

Để giám sát trạng thái ô kệ theo thời gian thực, bạn có thể thiết lập một vòng lặp định kỳ gọi API:

```python
import requests
import time

def monitor_slots(server_url="http://localhost:5000", interval=5):
    """
    Giám sát trạng thái ô kệ theo thời gian thực.
    
    Args:
        server_url: URL của server AI
        interval: Khoảng thời gian giữa các lần kiểm tra (giây)
    """
    previous_state = {}
    
    while True:
        try:
            response = requests.get(f"{server_url}/api/slots_status")
            
            if response.status_code == 200:
                data = response.json()
                signals = data.get("signals", [])
                
                # Tạo trạng thái hiện tại
                current_state = {s["id"]: s["has_goods"] for s in signals}
                
                # Phát hiện thay đổi
                if previous_state:
                    changes = []
                    for slot_id, has_goods in current_state.items():
                        if slot_id in previous_state and previous_state[slot_id] != has_goods:
                            changes.append({
                                "id": slot_id, 
                                "new_state": "có hàng" if has_goods else "trống",
                                "old_state": "có hàng" if previous_state[slot_id] else "trống"
                            })
                    
                    if changes:
                        print(f"\nPhát hiện {len(changes)} thay đổi:")
                        for change in changes:
                            print(f"  Ô #{change['id']}: {change['old_state']} -> {change['new_state']}")
                
                # Cập nhật trạng thái trước
                previous_state = current_state
                
                # Hiển thị tóm tắt
                filled_count = sum(1 for s in signals if s["has_goods"])
                empty_count = len(signals) - filled_count
                print(f"[{data['timestamp']}] Tổng số ô: {len(signals)}, Có hàng: {filled_count}, Trống: {empty_count}")
                
            elif response.status_code == 404:
                print("Chưa có dữ liệu phát hiện. Đang chờ cập nhật...")
            else:
                print(f"Lỗi: {response.status_code}")
        
        except Exception as e:
            print(f"Lỗi khi giám sát: {e}")
        
        time.sleep(interval)

if __name__ == "__main__":
    monitor_slots()
```

## 7. Tích hợp với server bên ngoài

Bạn có thể tích hợp API tín hiệu ô kệ với server bên ngoài bằng cách thiết lập một dịch vụ gọi API định kỳ và gửi dữ liệu đến server của bạn. Dưới đây là một ví dụ với Node.js:

```javascript
const axios = require('axios');
const express = require('express');

const app = express();
app.use(express.json());

// Lưu trữ trạng thái hiện tại
let currentSlotStatus = [];
let lastUpdateTime = '';

// Hàm lấy trạng thái ô kệ
async function fetchSlotStatus() {
  try {
    const response = await axios.get('http://localhost:5000/api/slots_status');
    
    if (response.data.success) {
      currentSlotStatus = response.data.signals;
      lastUpdateTime = response.data.timestamp;
      console.log(`Đã cập nhật trạng thái ô kệ: ${lastUpdateTime}`);
    }
  } catch (error) {
    console.error('Lỗi khi lấy trạng thái ô kệ:', error.message);
  }
}

// API endpoint để lấy trạng thái
app.get('/api/warehouse/slots', (req, res) => {
  res.json({
    timestamp: lastUpdateTime,
    slots: currentSlotStatus
  });
});

// Cập nhật trạng thái định kỳ
setInterval(fetchSlotStatus, 5000);

// Khởi động server
app.listen(3000, () => {
  console.log('Server đang chạy tại http://localhost:3000');
  // Cập nhật trạng thái lần đầu
  fetchSlotStatus();
});
```

## 8. Khắc phục sự cố

### Không có dữ liệu phát hiện

Nếu bạn nhận được lỗi "Chưa có dữ liệu phát hiện ô kệ", hãy chạy script `test_slot_detection.py` để cập nhật trạng thái:

```bash
python test_slot_detection.py ./images/warehouse_image.jpg
```

### Kết nối bị từ chối

Nếu bạn gặp lỗi "Connection refused", hãy đảm bảo rằng server AI đang chạy và lắng nghe trên cổng đúng:

```bash
python main.py --server
```

### Lỗi xử lý ảnh

Nếu phát hiện không chính xác, bạn có thể cần điều chỉnh thuật toán phát hiện trong hàm `_is_slot_filled` trong file `models/slot_detection_model.py` để phù hợp với điều kiện ánh sáng và hình ảnh cụ thể của kho hàng của bạn.