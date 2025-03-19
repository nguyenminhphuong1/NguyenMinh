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