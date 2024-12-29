from typing import Optional
import json
import mysql.connector


def init(config_path: Optional[str] = "/home/d_mao/.config/amazon_rds.json"):
    with open(config_path, mode='r') as f:
        db_config = json.load(f)
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    return conn, cursor
