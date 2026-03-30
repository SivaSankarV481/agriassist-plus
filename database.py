"""
database.py
============
MySQL database integration for AgriAssist+.
Gracefully skips all DB operations when MySQL is not configured,
so the app works on Streamlit Cloud without a database.
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ── Check whether MySQL is configured ────────────────────────────
_MYSQL_HOST = os.getenv("MYSQL_HOST", "").strip()
_MYSQL_ENABLED = bool(
    _MYSQL_HOST and
    os.getenv("MYSQL_USER", "").strip() and
    os.getenv("MYSQL_PASSWORD", "").strip()
)


class Database:
    def __init__(self):
        self._pool = None
        self.config = {
            "host":     os.getenv("MYSQL_HOST", "localhost"),
            "port":     int(os.getenv("MYSQL_PORT", 3306)),
            "user":     os.getenv("MYSQL_USER", "root"),
            "password": os.getenv("MYSQL_PASSWORD", ""),
            "database": os.getenv("MYSQL_DATABASE", "agriassist"),
        }
        if _MYSQL_ENABLED:
            self._connect()
        else:
            print("ℹ️  MySQL not configured — running without database (read-only mode).")

    def _connect(self):
        try:
            from mysql.connector import pooling
            self._pool = pooling.MySQLConnectionPool(
                pool_name="agriassist_pool",
                pool_size=5,
                **self.config,
            )
            print(f"✅ MySQL connected → {self.config['host']}:{self.config['port']}/{self.config['database']}")
        except Exception as e:
            print(f"⚠️  MySQL connection failed: {e}")
            self._pool = None

    def _get_conn(self):
        if self._pool is None:
            raise ConnectionError("MySQL pool not initialised.")
        return self._pool.get_connection()

    @property
    def is_connected(self) -> bool:
        return self._pool is not None

    def init_tables(self):
        if not self.is_connected:
            return
        ddl_statements = [
            """CREATE TABLE IF NOT EXISTS user_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL UNIQUE,
                district VARCHAR(100), crop VARCHAR(100),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_session (session_id)
            )""",
            """CREATE TABLE IF NOT EXISTS crop_price_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64), crop VARCHAR(100) NOT NULL,
                variety VARCHAR(100), district VARCHAR(100) NOT NULL,
                market VARCHAR(150), prediction_date DATE NOT NULL,
                current_price DECIMAL(10,2), price_1week DECIMAL(10,2),
                price_2week DECIMAL(10,2), price_1month DECIMAL(10,2),
                yield_loss_pct DECIMAL(5,2), yield_qty_qtl DECIMAL(10,2),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS weather_cache (
                id INT AUTO_INCREMENT PRIMARY KEY,
                district VARCHAR(100) NOT NULL,
                cache_key VARCHAR(64) NOT NULL UNIQUE,
                weather_json LONGTEXT NOT NULL,
                fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME NOT NULL,
                INDEX idx_expires (expires_at)
            )""",
            """CREATE TABLE IF NOT EXISTS knowledge_base_docs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                doc_id VARCHAR(50) NOT NULL UNIQUE,
                crop VARCHAR(100) NOT NULL, topic VARCHAR(100) NOT NULL,
                content LONGTEXT NOT NULL, source VARCHAR(200) DEFAULT 'built-in',
                is_active TINYINT(1) DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",
            """CREATE TABLE IF NOT EXISTS query_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(64), crop VARCHAR(100), district VARCHAR(100),
                query_type ENUM('diagnose','recommend','chemical','price') NOT NULL,
                user_query TEXT NOT NULL, ai_solution LONGTEXT,
                ai_recommendations LONGTEXT, weather_snapshot JSON,
                price_snapshot JSON, yield_loss_pct DECIMAL(5,2),
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )""",
        ]
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            for ddl in ddl_statements:
                cursor.execute(ddl)
            conn.commit()
            cursor.close(); conn.close()
            print("✅ All tables initialised.")
        except Exception as e:
            print(f"⚠️  Table init failed: {e}")

    def upsert_session(self, session_id: str, crop: str = "", district: str = ""):
        if not self.is_connected: return
        sql = """INSERT INTO user_sessions (session_id, crop, district) VALUES (%s,%s,%s)
                 ON DUPLICATE KEY UPDATE crop=VALUES(crop), district=VALUES(district), last_active=CURRENT_TIMESTAMP"""
        self._execute(sql, (session_id, crop, district))

    def save_prediction(self, session_id: str, data: dict) -> Optional[int]:
        if not self.is_connected: return None
        sql = """INSERT INTO crop_price_predictions
                 (session_id,crop,variety,district,market,prediction_date,
                  current_price,price_1week,price_2week,price_1month,yield_loss_pct,yield_qty_qtl)
                 VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        return self._execute_insert(sql, (
            session_id, data.get("crop"), data.get("variety"), data.get("district"),
            data.get("market"), data.get("prediction_date"), data.get("current_price"),
            data.get("price_1week"), data.get("price_2week"), data.get("price_1month"),
            data.get("yield_loss_pct"), data.get("yield_qty_qtl"),
        ))

    def get_cached_weather(self, district: str) -> Optional[dict]:
        if not self.is_connected: return None
        cache_key = hashlib.md5(district.encode()).hexdigest()
        rows = self._fetchall(
            "SELECT weather_json FROM weather_cache WHERE cache_key=%s AND expires_at>NOW() LIMIT 1",
            (cache_key,))
        if rows:
            try: return json.loads(rows[0]["weather_json"])
            except: return None
        return None

    def save_weather_cache(self, district: str, weather_data: dict, ttl_minutes: int = 60):
        if not self.is_connected: return
        cache_key = hashlib.md5(district.encode()).hexdigest()
        expires_at = datetime.now() + timedelta(minutes=ttl_minutes)
        sql = """INSERT INTO weather_cache (district,cache_key,weather_json,expires_at) VALUES(%s,%s,%s,%s)
                 ON DUPLICATE KEY UPDATE weather_json=VALUES(weather_json),
                 fetched_at=CURRENT_TIMESTAMP, expires_at=VALUES(expires_at)"""
        self._execute(sql, (district, cache_key, json.dumps(weather_data), expires_at))

    def sync_knowledge_base(self, docs: list):
        if not self.is_connected: return
        sql = """INSERT INTO knowledge_base_docs (doc_id,crop,topic,content,source) VALUES(%s,%s,%s,%s,'built-in')
                 ON DUPLICATE KEY UPDATE crop=VALUES(crop),topic=VALUES(topic),content=VALUES(content)"""
        for doc in docs:
            self._execute(sql, (doc["id"], doc["crop"], doc["topic"], doc["text"]))

    def save_query(self, session_id: str, data: dict) -> Optional[int]:
        if not self.is_connected: return None
        self.upsert_session(session_id, data.get("crop",""), data.get("district",""))
        sql = """INSERT INTO query_history
                 (session_id,crop,district,query_type,user_query,
                  ai_solution,ai_recommendations,weather_snapshot,price_snapshot,yield_loss_pct)
                 VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        return self._execute_insert(sql, (
            session_id, data.get("crop"), data.get("district"),
            data.get("query_type","recommend"), data.get("user_query",""),
            data.get("ai_solution"), data.get("ai_recommendations"),
            json.dumps(data.get("weather_snapshot")) if data.get("weather_snapshot") else None,
            json.dumps(data.get("price_snapshot"))   if data.get("price_snapshot")   else None,
            data.get("yield_loss_pct"),
        ))

    def test_connection(self) -> dict:
        try:
            rows = self._fetchall("SELECT VERSION() as version, NOW() as server_time")
            if rows:
                return {"ok": True, "version": rows[0]["version"], "server_time": str(rows[0]["server_time"])}
            return {"ok": False, "error": "No response"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _execute(self, sql, params=()):
        if not self.is_connected: return False
        try:
            conn = self._get_conn(); cursor = conn.cursor()
            cursor.execute(sql, params); conn.commit()
            cursor.close(); conn.close(); return True
        except Exception as e:
            print(f"⚠️  DB write error: {e}"); return False

    def _execute_insert(self, sql, params=()):
        if not self.is_connected: return None
        try:
            conn = self._get_conn(); cursor = conn.cursor()
            cursor.execute(sql, params); conn.commit()
            lid = cursor.lastrowid; cursor.close(); conn.close(); return lid
        except Exception as e:
            print(f"⚠️  DB insert error: {e}"); return None

    def _fetchall(self, sql, params=()):
        if not self.is_connected: return []
        try:
            conn = self._get_conn(); cursor = conn.cursor(dictionary=True)
            cursor.execute(sql, params); rows = cursor.fetchall()
            cursor.close(); conn.close(); return rows
        except Exception as e:
            print(f"⚠️  DB read error: {e}"); return []


_db_instance: Optional[Database] = None

def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
