# ============================================================
#  VIGILYNX — database.py
#  Fixed: head_turn count was 0 due to old field name mismatch
#  Now queries both 'type' AND 'event_type' fields
#  Statuses: unreviewed | reviewed | actioned
# ============================================================

from datetime import datetime

_collection = None
_in_memory  = []
_id_counter = [0]

try:
    from pymongo import MongoClient
    _client     = MongoClient('mongodb://localhost:27017/',
                              serverSelectionTimeoutMS=2000)
    _client.server_info()
    _db         = _client['vigilynx']
    _collection = _db['incidents']
    print("[DB] MongoDB connected → vigilynx.incidents")

    # ── FIX OLD RECORDS ──────────────────────────────────────
    # Old database.py saved field as 'event_type'
    # New one saves as 'type'
    # This migrates old records automatically on startup
    try:
        fixed = _collection.update_many(
            {'event_type': {'$exists': True}, 'type': {'$exists': False}},
            [{'$set': {'type': '$event_type'}}]
        )
        if fixed.modified_count > 0:
            print(f"[DB] Migrated {fixed.modified_count} old records (event_type → type)")
    except Exception:
        pass

except Exception:
    print("[DB] MongoDB not found — using in-memory store")

# Only 3 statuses
VALID_STATUSES = ['unreviewed', 'reviewed', 'actioned', 'safe', 'fine_imposed', 'pending']


def save_incident(event_type, bbox=None, conf=0.0, img_path=None):
    _id_counter[0] += 1
    now = datetime.now()
    doc = {
        'id'         : _id_counter[0],
        'type'       : event_type,       # always save as 'type'
        'bbox'       : list(bbox) if bbox else None,
        'conf'       : round(float(conf), 4),
        'img_path'   : img_path,
        'timestamp'  : now.isoformat(),
        'ts'         : now.strftime('%H:%M:%S'),
        'date'       : now.strftime('%Y-%m-%d'),
        'status'     : 'unreviewed',
        'reviewed_by': None,
        'reviewed_at': None,
    }
    if _collection is not None:
        try:
            result = _collection.insert_one(doc)
            doc['_id'] = str(result.inserted_id)
            return doc
        except Exception as e:
            print(f"[DB] Save failed: {e}")
    _in_memory.append(doc)
    return doc


def update_status(incident_id, new_status, reviewed_by='Authority'):
    if new_status not in VALID_STATUSES:
        return False
    now = datetime.now().isoformat()
    db_ok = False
    if _collection is not None:
        try:
            result = _collection.update_one(
                {'id': int(incident_id)},
                {'$set': {
                    'status'     : new_status,
                    'reviewed_by': reviewed_by,
                    'reviewed_at': now,
                }}
            )
            db_ok = result.matched_count > 0
        except Exception as e:
            print(f"[DB] Update failed: {e}")
    # Always sync in-memory store regardless of MongoDB result
    for doc in _in_memory:
        if str(doc.get('id')) == str(incident_id):
            doc['status']      = new_status
            doc['reviewed_by'] = reviewed_by
            doc['reviewed_at'] = now
            return True
    return db_ok


def get_recent_incidents(limit=100):
    if _collection is not None:
        try:
            cursor = (_collection.find({}, {'_id': 0})
                                  .sort('timestamp', -1)
                                  .limit(limit))
            return list(cursor)
        except Exception as e:
            print(f"[DB] Read failed: {e}")
    return list(reversed(_in_memory[-limit:]))


def get_stats():
    types = ['phone', 'head_turn', 'crowd']
    if _collection is not None:
        try:
            total = _collection.count_documents({})
            stats = {'total': total}
            for t in types:
                # Query BOTH field names for full compatibility
                stats[t] = _collection.count_documents({
                    '$or': [{'type': t}, {'event_type': t}]
                })
            for s in VALID_STATUSES:
                stats[s] = _collection.count_documents({'status': s})
            return stats
        except Exception:
            pass
    # In-memory fallback
    stats = {'total': len(_in_memory)}
    for t in types:
        stats[t] = sum(
            1 for d in _in_memory
            if d.get('type') == t or d.get('event_type') == t
        )
    for s in VALID_STATUSES:
        stats[s] = sum(1 for d in _in_memory if d.get('status') == s)
    return stats
