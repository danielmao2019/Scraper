from typing import List, Dict, Any, Optional
import json
import psycopg2
import utils


def _serialize(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serializes a record's list or dictionary fields to JSON strings.
    """
    return {
        key: json.dumps(record[key]) if isinstance(record[key], (list, dict)) else record[key]
        for key in record
    }


def insert_record(
    record: Dict[str, Any],
    cursor: Optional[psycopg2.extensions.cursor] = None,
) -> None:
    """
    Inserts a single record into the `papers` table.
    """
    conn = None
    should_cleanup = cursor is None
    if should_cleanup:
        conn, cursor = utils.db.init()

    record = _serialize(record)

    check_query = """
    SELECT * FROM papers WHERE id = %s OR title = %s OR %s = ANY((urls->>'html')::text[])
    """
    cursor.execute(check_query, (record['id'], record['title'], record['html_url']))
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
        record['id'], record['code_names'], record['title'], json.dumps(record['urls']),
        record['pub_name'], record['pub_date'], json.dumps(record['authors']), record['abstract'],
        record['full_text'], json.dumps(record['comments']),
    ))

    if should_cleanup and conn:
        conn.commit()
        cursor.close()
        conn.close()


def insert_records(records: List[Dict[str, Any]], cursor: Optional[psycopg2.extensions.cursor] = None) -> None:
    """
    Inserts multiple records into the `papers` table.
    """
    conn = None
    should_cleanup = cursor is None
    if should_cleanup:
        conn, cursor = utils.db.init()

    for record in records:
        insert_record(record, cursor)

    if should_cleanup and conn:
        conn.commit()
        cursor.close()
        conn.close()
