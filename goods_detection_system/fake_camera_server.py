import cv2
import time
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# Đường dẫn đến thư mục chứa ảnh
image_dir = "/home/admin/CameraAi/goods_detection_system/images"
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]


class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()

        try:
            while True:
                for img_file in image_files:
                    img_path = os.path.join(image_dir, img_file)
                    img = cv2.imread(img_path)

                    if img is not None:
                        # Thêm text hiển thị tên file
                        cv2.putText(img, img_file, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Mã hóa ảnh thành JPEG
                        _, jpeg = cv2.imencode('.jpg', img)

                        # Gửi ảnh
                        self.wfile.write(b"--jpgboundary\r\n")
                        self.wfile.write(b"Content-type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b"\r\n")

                        # Đợi một chút trước khi chuyển sang ảnh tiếp theo
                        time.sleep(2)

        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnect

    def log_message(self, format, *args):
        # Vô hiệu hóa log để tránh rối loạn dòng lệnh
        return


# Chạy server trên một thread riêng
def run_server():
    server = HTTPServer(('localhost', 8080), SimpleHTTPRequestHandler)
    print(f"Đã khởi động server giả lập camera tại http://localhost:8080")
    server.serve_forever()


# Chạy server
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

print("Server đang chạy... Cấu hình camera IP trong ứng dụng của bạn với URL:")
print("http://localhost:8080")
print("Nhấn Ctrl+C để dừng.")

try:
    # Giữ chương trình chạy
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Đang dừng...")