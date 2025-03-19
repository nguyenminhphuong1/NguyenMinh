#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
import json
import platform
import time
import threading
import multiprocessing
from typing import Dict, List, Any, Optional

# Đặt đường dẫn gốc
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

# Import các module cần thiết
from config.settings import load_camera_config, AI_SERVER, NOTIFICATION_SERVER, UI_CONFIG
from utils.camera_manager import CameraManager
from utils.notification import NotificationManager
from server.ai_server import AIServer
from frontend.gui import MainWindow

# Import module PyQt5 cho giao diện
from PyQt5.QtWidgets import QApplication

# Thiết lập logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(ROOT_DIR, 'logs', 'app.log'), mode='a')
    ]
)
logger = logging.getLogger("main")


def ensure_directories():
    """Đảm bảo các thư mục cần thiết tồn tại."""
    dirs = [
        os.path.join(ROOT_DIR, 'logs'),
        os.path.join(ROOT_DIR, 'models', 'weights'),
        os.path.join(ROOT_DIR, 'data')
    ]

    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Đã tạo thư mục: {directory}")


def start_ai_server(config: Dict):
    """Khởi động server AI trong một tiến trình riêng."""
    server = AIServer(config)
    server.start(debug=False, threaded=True)


def main():
    """Hàm chính của ứng dụng."""
    # Phân tích đối số dòng lệnh
    parser = argparse.ArgumentParser(description="Hệ thống phát hiện hàng hóa qua camera sử dụng Fast R-CNN")
    parser.add_argument("--gui", action="store_true", help="Khởi động giao diện người dùng")
    parser.add_argument("--server", action="store_true", help="Khởi động server AI")
    parser.add_argument("--debug", action="store_true", help="Chạy ở chế độ debug")
    args = parser.parse_args()

    # Đảm bảo các thư mục cần thiết tồn tại
    ensure_directories()

    # Hiển thị thông tin hệ thống
    logger.info(f"Hệ thống phát hiện hàng hóa qua camera sử dụng Fast R-CNN")
    logger.info(f"Hệ điều hành: {platform.system()} {platform.release()}")
    logger.info(f"Python: {platform.python_version()}")

    # Tải cấu hình camera
    camera_config = load_camera_config()

    # Nếu không có đối số, mặc định chạy cả GUI và server
    if not (args.gui or args.server):
        args.gui = True
        args.server = True

    # Khởi động server AI nếu cần
    server_process = None
    if args.server:
        logger.info("Đang khởi động server AI...")
        server_process = multiprocessing.Process(
            target=start_ai_server,
            args=(AI_SERVER,),
            daemon=True
        )
        server_process.start()
        logger.info(f"Server AI đã được khởi động (PID: {server_process.pid})")

        # Chờ server khởi động
        time.sleep(2)

    # Khởi động GUI nếu cần
    if args.gui:
        logger.info("Đang khởi động giao diện người dùng...")

        # Tạo ứng dụng QApplication
        app = QApplication(sys.argv)

        # Thiết lập cấu hình cho main window
        config = {
            "ui_config": UI_CONFIG,
            "camera_config": camera_config,
            "ai_server": AI_SERVER,
            "notification_server": NOTIFICATION_SERVER
        }

        # Tạo và hiển thị cửa sổ chính
        main_window = MainWindow(config)
        main_window.show()

        logger.info("Giao diện người dùng đã được khởi động")

        # Chạy vòng lặp sự kiện
        sys.exit(app.exec_())

    # Nếu chỉ chạy server mà không có GUI, chờ đến khi người dùng bấm Ctrl+C
    if args.server and not args.gui:
        try:
            logger.info("Server đang chạy. Nhấn Ctrl+C để dừng.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Đã nhận tín hiệu dừng. Đang dọn dẹp...")
        finally:
            if server_process and server_process.is_alive():
                server_process.terminate()
                server_process.join(timeout=5)
                logger.info("Server AI đã được dừng")


if __name__ == "__main__":
    # Đảm bảo đặt đúng thư mục làm việc
    os.chdir(ROOT_DIR)

    # Bắt đầu ứng dụng
    main()