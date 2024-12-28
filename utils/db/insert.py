from typing import List, Dict, Any
import json
import mysql.connector
import utils


def _serialize(record: Dict[str, Any]) -> Dict[str, str]:
    return {
        key: json.dumps(record[key]) if type(record[key]) in [list, dict] else record[key]
        for key in record
    }


def _insert_single(cursor, record: Dict[str, str]) -> None:
    record = _serialize(record)
    check_query = """
SELECT * FROM papers WHERE id = %s OR title = %s
    """
    cursor.execute(check_query, (record['id'], record['title']))
    existing = cursor.fetchone()
    if existing:
        print(f"Duplicate detected for ID: {record['id']} or Title: {record['title']}")
        print("Existing record:")
        print(json.dumps(existing, indent=4))
        print("New record:")
        print(json.dumps(record, indent=4))
        return
    insert_query = """
INSERT INTO papers (
    id, code_names, title, urls, pub_name, pub_date, authors, 
    abstract, full_text, comments
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (
        record['id'], record['code_names'], record['title'], record['urls'],
        record['pub_name'], record['pub_date'], record['authors'], record['abstract'],
        record['full_text'], record['comments'],
    ))


def insert(records: List[Dict[str, str]]) -> None:
    try:
        conn, cursor = utils.db.init()
        for record in records:
            _insert_single(cursor, record)
        conn.commit()
        print("All valid records have been inserted.")
    except mysql.connector.Error as err:
        print(f"Error: {err}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
