# ============================================================
#  VIGILYNX — database.py
#  MongoDB with in-memory fallback.
#  Supports full incident review workflow.
# ============================================================

from datetime import datetime

_collection  = None
_in_memory   = []
_id_counter  = [0]

try:
    from pymongo import MongoClient
    _client     = MongoClient('mongodb://localhost:27017/',
                              serverSelectionTimeoutMS=2000)
    _client.server_info()
    _db         = _client['vigilynx']
    _collection = _db['incidents']
    print("[DB] MongoDB connected → vigilynx.incidents")
except Exception:
    print("[DB] MongoDB not available — using in-memory store")

VALID_STATUSES = ['unreviewed', 'safe', 'action_taken', 'pending']


def save_incident(event_type, bbox=None, conf=0.0, img_path=None):
    _id_counter[0] += 1
    doc = {
        'id'          : _id_counter[0],
        'type'        : event_type,
        'bbox'        : list(bbox) if bbox else None,
        'conf'        : round(float(conf), 4),
        'img_path'    : img_path,
        'timestamp'   : datetime.now().isoformat(),
        'ts'          : datetime.now().strftime('%H:%M:%S'),
        'date'        : datetime.now().strftime('%Y-%m-%d'),
        'status'      : 'unreviewed',
        'review_note' : '',
        'reviewed_by' : None,
        'reviewed_at' : None,
    }
    if _collection is not None:
        try:
            _collection.insert_one(doc)
            doc.pop('_id', None)
        except Exception:
            pass
    _in_memory.append(doc)
    return doc


def update_incident_review(incident_id, status, review_note, reviewed_by='Authority'):
    """Update status + review note for an incident."""
    if status not in VALID_STATUSES:
        return False
    now = datetime.now().isoformat()
    update = {
        'status'      : status,
        'review_note' : review_note,
        'reviewed_by' : reviewed_by,
        'reviewed_at' : now,
    }
    if _collection is not None:
        try:
            result = _collection.update_one(
                {'id': int(incident_id)}, {'$set': update})
            if result.modified_count > 0:
                return True
        except Exception:
            pass
    for doc in _in_memory:
        if str(doc.get('id')) == str(incident_id):
            doc.update(update)
            return True
    return False


def get_recent_incidents(limit=50):
    if _collection is not None:
        try:
            cursor = (_collection
                      .find({}, {'_id': 0})
                      .sort('timestamp', -1)
                      .limit(limit))
            return list(cursor)
        except Exception:
            pass
    return list(reversed(_in_memory[-limit:]))


def get_stats():
    if _collection is not None:
        try:
            total = _collection.count_documents({})
            return {
                'total'       : total,
                'phone'       : _collection.count_documents({'type': 'phone'}),
                'head_turn'   : _collection.count_documents({'type': 'head_turn'}),
                'crowd'       : _collection.count_documents({'type': 'crowd'}),
                'unreviewed'  : _collection.count_documents({'status': 'unreviewed'}),
                'safe'        : _collection.count_documents({'status': 'safe'}),
                'action_taken': _collection.count_documents({'status': 'action_taken'}),
                'pending'     : _collection.count_documents({'status': 'pending'}),
            }
        except Exception:
            pass
    return {
        'total'       : len(_in_memory),
        'phone'       : sum(1 for d in _in_memory if d['type'] == 'phone'),
        'head_turn'   : sum(1 for d in _in_memory if d['type'] == 'head_turn'),
        'crowd'       : sum(1 for d in _in_memory if d['type'] == 'crowd'),
        'unreviewed'  : sum(1 for d in _in_memory if d.get('status') == 'unreviewed'),
        'safe'        : sum(1 for d in _in_memory if d.get('status') == 'safe'),
        'action_taken': sum(1 for d in _in_memory if d.get('status') == 'action_taken'),
        'pending'     : sum(1 for d in _in_memory if d.get('status') == 'pending'),
    }
