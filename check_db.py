"""
check_db.py
============
Run this to diagnose MySQL connection issues.
    python check_db.py
"""

print("=" * 50)
print("  AgriAssist+ MySQL Diagnostic")
print("=" * 50)

# ── Step 1: Check dotenv ──────────────────────────────
print("\n[1] Checking .env file...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("    ✅ python-dotenv loaded")
except ImportError:
    print("    ❌ python-dotenv not installed → run: pip install python-dotenv")
    exit(1)

import os
host     = os.getenv("MYSQL_HOST", "NOT SET")
port     = os.getenv("MYSQL_PORT", "NOT SET")
user     = os.getenv("MYSQL_USER", "NOT SET")
password = os.getenv("MYSQL_PASSWORD", "NOT SET")
database = os.getenv("MYSQL_DATABASE", "NOT SET")

print(f"    HOST     = {host}")
print(f"    PORT     = {port}")
print(f"    USER     = {user}")
print(f"    PASSWORD = {'*' * len(password) if password != 'NOT SET' else 'NOT SET'}")
print(f"    DATABASE = {database}")

if "NOT SET" in [host, user, password, database]:
    print("\n    ❌ One or more MySQL env vars are missing in your .env file!")
    exit(1)

# ── Step 2: Check mysql-connector-python ─────────────
print("\n[2] Checking mysql-connector-python...")
try:
    import mysql.connector
    print(f"    ✅ mysql-connector-python installed (version: {mysql.connector.__version__})")
except ImportError:
    print("    ❌ Not installed → run: pip install mysql-connector-python")
    exit(1)

# ── Step 3: Try connecting WITHOUT selecting a database ──
print("\n[3] Testing connection to MySQL server (no database)...")
try:
    conn = mysql.connector.connect(
        host=host,
        port=int(port),
        user=user,
        password=password,
    )
    cursor = conn.cursor()
    cursor.execute("SELECT VERSION()")
    version = cursor.fetchone()[0]
    print(f"    ✅ Connected! MySQL version: {version}")
    cursor.close()
    conn.close()
except mysql.connector.Error as e:
    print(f"    ❌ Connection failed: {e}")
    print("\n    Common fixes:")
    print("    - Make sure MySQL service is running (check Services in Task Manager)")
    print("    - Verify your password is correct")
    print("    - Try: mysql -u root -p  in your terminal to test manually")
    exit(1)

# ── Step 4: Check if database exists, create if not ──
print(f"\n[4] Checking if database '{database}' exists...")
try:
    conn = mysql.connector.connect(
        host=host,
        port=int(port),
        user=user,
        password=password,
    )
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES")
    dbs = [row[0] for row in cursor.fetchall()]
    if database in dbs:
        print(f"    ✅ Database '{database}' already exists")
    else:
        print(f"    ⚠️  Database '{database}' not found — creating it...")
        cursor.execute(f"CREATE DATABASE {database} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        print(f"    ✅ Database '{database}' created!")
    cursor.close()
    conn.close()
except mysql.connector.Error as e:
    print(f"    ❌ Failed: {e}")
    exit(1)

# ── Step 5: Connect with database selected ────────────
print(f"\n[5] Connecting to '{database}' database...")
try:
    conn = mysql.connector.connect(
        host=host,
        port=int(port),
        user=user,
        password=password,
        database=database,
    )
    print("    ✅ Connected to database successfully")
    conn.close()
except mysql.connector.Error as e:
    print(f"    ❌ Failed: {e}")
    exit(1)

# ── Step 6: Run full database.py init ─────────────────
print("\n[6] Running database.py init_tables()...")
try:
    from database import get_db
    db = get_db()
    if db.is_connected:
        db.init_tables()
        print("    ✅ All tables created successfully")
    else:
        print("    ❌ Database singleton failed to connect")
        exit(1)
except Exception as e:
    print(f"    ❌ Error: {e}")
    exit(1)

# ── Step 7: Sync knowledge base ───────────────────────
print("\n[7] Syncing knowledge base to MySQL...")
try:
    from rag_engine import KNOWLEDGE_BASE
    db.sync_knowledge_base(KNOWLEDGE_BASE)
    print(f"    ✅ Synced {len(KNOWLEDGE_BASE)} docs")
except Exception as e:
    print(f"    ❌ Error: {e}")

# ── Step 8: Quick read test ───────────────────────────
print("\n[8] Running read test...")
try:
    result = db.test_connection()
    if result["ok"]:
        print(f"    ✅ Read test passed — server time: {result['server_time']}")
    else:
        print(f"    ❌ Read test failed: {result['error']}")
except Exception as e:
    print(f"    ❌ Error: {e}")

print("\n" + "=" * 50)
print("  ✅ All checks passed! MySQL is ready.")
print("  You can now run: streamlit run app.py")
print("=" * 50)
