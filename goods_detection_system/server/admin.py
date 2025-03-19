# Tạo file server/admin.py
import os
import flask
from flask import Blueprint, render_template, request, redirect, url_for, jsonify
import requests

# Tạo blueprint cho trang quản trị
admin_bp = Blueprint('admin', __name__, template_folder='templates')

@admin_bp.route('/')
def index():
    """Trang chủ quản trị."""
    return render_template('admin/index.html')

@admin_bp.route('/data')
def data_management():
    """Quản lý dữ liệu huấn luyện."""
    return render_template('admin/data.html')

@admin_bp.route('/models')
def model_management():
    """Quản lý mô hình."""
    return render_template('admin/models.html')

@admin_bp.route('/training')
def training():
    """Huấn luyện mô hình."""
    return render_template('admin/training.html')

# Đăng ký blueprint trong ai_server.py
# Thêm dòng sau vào phương thức _setup_routes trong class AIServer
# self.app.register_blueprint(admin_bp, url_prefix='/admin')