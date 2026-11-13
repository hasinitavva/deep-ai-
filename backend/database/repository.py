import json
from backend.database.db import get_db_connection

def save_prediction(filename, primary_result, total_faces=1):
    """
    Save prediction entry to SQLite database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    explanation_str = json.dumps(primary_result.get('explanation_reasons', []))
    heatmap_url = primary_result.get('heatmap_urls', {}).get('overlay', '')
    
    cursor.execute('''
        INSERT INTO predictions (
            filename, prediction, confidence, face_count,
            dominant_region, frequency_score, explanation_json, heatmap_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        filename,
        primary_result.get('prediction', 'UNKNOWN'),
        primary_result.get('confidence', 0.0),
        total_faces,
        primary_result.get('dominant_region', 'N/A'),
        primary_result.get('frequency_anomaly_score', 0.0),
        explanation_str,
        heatmap_url
    ))
    
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    return inserted_id

def fetch_history(limit=50):
    """
    Fetch recent prediction history records.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, filename, prediction, confidence, face_count,
               dominant_region, frequency_score, explanation_json, heatmap_path, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            'id': r['id'],
            'filename': r['filename'],
            'prediction': r['prediction'],
            'confidence': r['confidence'],
            'face_count': r['face_count'],
            'dominant_region': r['dominant_region'],
            'frequency_score': r['frequency_score'],
            'explanation_reasons': json.loads(r['explanation_json']) if r['explanation_json'] else [],
            'heatmap_path': r['heatmap_path'],
            'created_at': r['created_at']
        })
    return history
