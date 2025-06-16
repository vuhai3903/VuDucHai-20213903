from fastapi import FastAPI
from database import get_connection

app = FastAPI()

@app.get("/")
def read_attacks():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM attack_sessions ORDER BY id DESC LIMIT 15")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return {"data": rows}
