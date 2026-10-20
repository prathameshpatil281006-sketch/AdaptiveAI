import sqlite3

c = sqlite3.connect("data/adaptive_ai.db")

tables = c.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
).fetchall()

print(tables)

c.close()