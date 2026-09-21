import pandas as pd, psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
df = pd.read_csv("data/synthetic/workers.csv")
with conn.cursor() as cur:
    for _, row in df.iterrows():
        skills = row["skills"].split(";") if pd.notna(row["skills"]) else []
        cur.execute("UPDATE workers SET skills = %s WHERE worker_id = %s", (skills, row["worker_id"]))
conn.commit()