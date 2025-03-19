#!/bin/bash
# setup.sh - Script cài đặt tự động cho hệ thống phát hiện hàng hóa

echo "==== Bắt đầu cài đặt hệ thống phát hiện hàng hóa ===="

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate

# Cài đặt các gói phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt

# Tạo các thư mục cần thiết
mkdir -p models/weights
mkdir -p data/goods_detection/images
mkdir -p data/goods_detection/annotations
mkdir -p logs

echo "==== Cài đặt hoàn tất ===="
echo "Để kích hoạt môi trường, chạy: source venv/bin/activate"
echo "Để khởi động ứng dụng, chạy: python main.py"