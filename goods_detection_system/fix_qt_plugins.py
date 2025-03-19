import os
import sys
import cv2
import shutil

# Tìm đường dẫn đến thư mục plugins của OpenCV
cv2_path = os.path.dirname(cv2.__file__)
qt_plugin_path = os.path.join(cv2_path, 'qt', 'plugins')

if os.path.exists(qt_plugin_path):
    # Đổi tên thư mục plugins của OpenCV
    backup_path = qt_plugin_path + '.bak'
    print(f"Đổi tên {qt_plugin_path} thành {backup_path}")
    if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
    shutil.move(qt_plugin_path, backup_path)
    print("Hoàn tất!")
else:
    print(f"Thư mục {qt_plugin_path} không tồn tại.")