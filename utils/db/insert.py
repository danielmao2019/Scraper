from typing import List, Dict, Any, Optional
import json
import mysql.connector
import utils


def _serialize(record: Dict[str, Any]) -> Dict[str, str]:
    return {
        key: json.dumps(record[key]) if type(record[key]) in [list, dict] else record[key]
        for key in record
    }


def insert_record(
    record: Dict[str, str],
    cursor: Optional[mysql.connector.cursor_cext.CMySQLCursor] = None,
) -> None:
    if cursor is None:
        conn, cursor = utils.db.init()
        should_cleanup: bool = True
    else:
        should_cleanup: bool = False
    record = _serialize(record)
    check_query = """
SELECT * FROM papers WHERE id = %s OR title = %s OR JSON_CONTAINS(urls->'$.html', %s)
    """
    cursor.execute(check_query, (record['id'], record['title'], json.dumps(record['html_url'])))
    existing = cursor.fetchone()
    cursor.fetchall()
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
    cursor.fetchall()
    if should_cleanup:
        conn.commit()
        cursor.close()
        conn.close()


def insert_records(records: List[Dict[str, str]], cursor: Optional[mysql.connector.cursor_cext.CMySQLCursor]) -> None:
    if cursor is None:
        conn, cursor = utils.db.init()
        should_cleanup: bool = True
    else:
        should_cleanup: bool = False
    for record in records:
        insert_record(record=record, cursor=cursor)
    if should_cleanup:
        conn.commit()
        cursor.close()
        conn.close()
