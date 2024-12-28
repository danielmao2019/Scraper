from typing import Optional
import json
import mysql.connector


def init(config_path: Optional[str] = "/home/d_mao/.config/amazon_rds.json"):
    with open(config_path, mode='r') as f:
        db_config = json.load(f)
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    return conn, cursor


def execute(conn, cursor, query: str) -> None:
    try:
        cursor.execute(query)
        conn.commit()
        print("Table `papers` created successfully (if it did not already exist).")

    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
