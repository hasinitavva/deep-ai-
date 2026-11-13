import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), 'history.db')

def get_db_connection():
    """
    Establish connection to SQLite database with Row factory enabled.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initialize SQLite tables for prediction history log.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            face_count INTEGER NOT NULL,
            dominant_region TEXT,
            frequency_score REAL,
            explanation_json TEXT,
            heatmap_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"[Database] SQLite DB initialized at '{DB_PATH}'")

if __name__ == '__main__':
    init_db()
