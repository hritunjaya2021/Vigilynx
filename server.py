# server.py — Render deployment (no camera, API + dashboard only)

from flask import Flask, jsonify, request
from flask_cors import CORS
from database import save_incident, get_recent_incidents, get_stats, update_status

flask_app = Flask(__name__)
CORS(flask_app)

from flask import Flask, jsonify, request, send_file
...

@flask_app.route('/')
def index():
    return send_file('vigilynx_website.html')

@flask_app.route('/api/alerts')
def api_alerts():
    return jsonify(get_recent_incidents(100))

@flask_app.route('/api/stats')
def api_stats():
    return jsonify(get_stats())

@flask_app.route('/api/update_status', methods=['POST', 'OPTIONS'])
def api_update_status():
    if request.method == 'OPTIONS':
        return '', 200
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'success': False, 'error': 'No JSON body'}), 400
    ok = update_status(data.get('id'), data.get('status'))
    return jsonify({'success': ok})

@flask_app.route('/video_feed')
def video_feed():
    return jsonify({'error': 'Camera not available on server'}), 503

app = flask_app

if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=5000)
