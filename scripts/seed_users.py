import sqlite3

DB_FILE = "data/users.db"

USERS = [
    ("alice", "alice123"),
    ("bob", "bob123"),
    ("carol", "carol123"),
    ("dave", "dave123"),
    ("erin", "erin123"),
    ("frank", "frank123"),
    ("grace", "grace123"),
    ("heidi", "heidi123"),
    ("ivan", "ivan123"),
    ("judy", "judy123"),
    ("mallory", "mallory123"),
    ("niaj", "niaj123"),
]

conn = sqlite3.connect(DB_FILE)
conn.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
conn.executemany("INSERT OR REPLACE INTO users (username, password) VALUES (?, ?)", USERS)
conn.commit()
conn.close()

print(f"Seeded {len(USERS)} users into {DB_FILE}")
