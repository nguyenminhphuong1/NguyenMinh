from flask import Flask, request, jsonify
import logging
import json

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


@app.route('/notification', methods=['POST'])
def receive_notification():
    data = request.json
    logging.info(f"Đã nhận thông báo loại: {data.get('type')}")
    logging.info(f"Thời gian: {data.get('timestamp')}")
    logging.info(f"Camera: {data.get('camera_id')}")

    # In thông tin chi tiết về các ô kệ
    slots_data = data.get('slots_data', [])
    logging.info(f"Tổng số ô kệ: {len(slots_data)}")

    filled_count = sum(1 for slot in slots_data if slot.get('has_goods', False))
    empty_count = len(slots_data) - filled_count

    logging.info(f"Số ô có hàng: {filled_count}")
    logging.info(f"Số ô trống: {empty_count}")

    # In thông tin từng ô kệ
    for slot in slots_data:
        slot_id = slot.get('slot_id', 'N/A')
        status = "Có hàng" if slot.get('has_goods', False) else "Trống"
        pos = slot.get('position', {})
        row = pos.get('row', 'N/A')
        col = pos.get('column', 'N/A')

        logging.info(f"Ô #{slot_id} (dãy {row}, cột {col}): {status}")

    return jsonify({
        "success": True,
        "message": f"Đã nhận thông báo cho {len(slots_data)} ô kệ"
    })


if __name__ == '__main__':
    print("Server nhận thông báo đang chạy tại http://localhost:8000")
    app.run(host='localhost', port=9000)