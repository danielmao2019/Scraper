from typing import List, Dict, Any
import os
import json
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
import mysql.connector
import scrape
import utils


def scrape_or_query(url: str, cursor: mysql.connector.cursor_cext.CMySQLCursor) -> Dict[str, Any]:
    result = {key: None for key in [
        'id', 'code_names', 'title', 'urls',
        'pub_name', 'pub_date', 'authors', 'abstract',
        'full_text', 'comments',
    ]}
    result['id'] = ""
    
    # Query to check if the HTML URL exists in the `urls.html` array
    same_html_query = """
    SELECT * 
    FROM `papers`
    WHERE JSON_CONTAINS(urls->'$.html', %s)
    """
    cursor.execute(same_html_query, (json.dumps(url),))  # Serialize the URL as JSON
    same_html_record = cursor.fetchall()
    
    if same_html_record:
        # If the HTML URL already exists, return the record
        assert len(same_html_record) == 1, f"{same_html_record=}"
        result.update(same_html_record[0])
    else:
        # If no record exists with the same HTML URL, scrape the new record
        scraped_record: Dict[str, str] = scrape.scrape(url, compile=False)
        title = scraped_record['title']
        
        # Check if a record with the same title exists
        same_title_query = """
        SELECT * 
        FROM `papers`
        WHERE title = %s
        """
        cursor.execute(same_title_query, (title,))
        same_title_record = cursor.fetchall()
        
        if same_title_record:
            # If a record with the same title exists, update its `urls` field
            assert len(same_title_record) == 1, f"{same_title_record=}"
            same_title_record = same_title_record[0]
            urls = json.loads(same_title_record['urls'])  # Load the existing URLs as a Python dict
            # Add the new URLs to the respective arrays, ensuring no duplicates
            urls['html'] = urls['html'] + [scraped_record['html_url']]
            urls['pdf'] = urls['pdf'] + [scraped_record['pdf_url']]
            result.update(same_title_record)
            result['urls'] = urls  # Reflect the updated URLs in the result

            # Update the `urls` field in the database
            update_query = """
            UPDATE `papers`
            SET urls = %s
            WHERE title = %s
            """
            cursor.execute(update_query, (json.dumps(urls), same_title_record['title']))
        else:
            # Insert the new record if no matching title is found
            result.update(scraped_record)
            insert_query = """
            INSERT INTO `papers` (
                id, code_names, title, urls, pub_name, pub_date, authors, 
                abstract, full_text, comments
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, (
                result['id'], json.dumps(result['code_names']), result['title'],
                json.dumps(result['urls']), result['pub_name'], result['pub_date'],
                json.dumps(result['authors']), result['abstract'],
                result['full_text'], json.dumps(result['comments']),
            ))
    
    return scrape.utils.compile_markdown(**result)


def main(urls: List[str]):
    """
    Arguments:
        urls (List[str]): A list of html urls to scrape from.
    """
    assert type(urls) == list, f"{type(urls)=}"
    assert all(map(lambda x: type(x) == str, urls))
    urls_unique: List[str] = []
    for url in urls:
        if url not in urls_unique:
            urls_unique.append(url)
    if len(urls_unique) != len(urls):
        logging.info(f"Provided list contains duplicates. Reduced to {len(urls_unique)} papers.")
    # connect to db
    conn, cursor = utils.db.init()
    # iterate through unique url list and scrape each
    filepath = os.path.join("scrape", "scraped_papers.md")
    with open(filepath, mode='w', encoding='utf8') as f:
        for idx, url in enumerate(urls_unique):
            logging.info(f"[{idx+1}/{len(urls_unique)}] Scraping {url}")
            f.write(scrape_or_query(url=url, cursor=cursor))
    conn.commit()
    if conn.is_connected():
        cursor.close()
        conn.close()
    logging.info(f"Process terminated.")


if __name__ == "__main__":
    from scrape.papers_to_scrape import papers_to_scrape
    main(urls=papers_to_scrape)
