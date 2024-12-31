from typing import Tuple, List, Dict, Union, Optional
import os
import json
import jsbeautifier
from concurrent.futures import ThreadPoolExecutor
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
import mysql.connector
import scrape
import search
import utils


def scrape_or_query(url: str, cursor: mysql.connector.cursor_cext.CMySQLCursor) -> str:
    result = {key: None for key in [
        'id', 'code_names', 'title', 'urls',
        'pub_name', 'pub_date', 'authors', 'abstract',
        'full_text', 'comments',
    ]}
    result['id'] = ""

    # Query to check if the HTML URL exists in the `urls.html` array
    same_html_query = """
    SELECT id
    FROM `papers`
    WHERE JSON_CONTAINS(urls->'$.html', %s)
    """
    same_html_record = utils.db.execute(cursor, same_html_query, (json.dumps(url),))

    if same_html_record:
        # If the HTML URL already exists, return the record
        assert len(same_html_record) == 1, f"{same_html_record=}"
        same_html_record = same_html_record[0]
        result.update(same_html_record)
        if same_html_record['full_text'] is None:
            urls = json.loads(same_html_record['urls'])
            urls = list(zip(urls['html'], urls['pdf']))
            urls = list(filter(lambda x: x[0] == url, urls))
            assert len(urls) == 1, f"{urls=}"
            try:
                result['full_text'] = search.extract_text(urls[0][1])
            except Exception as e:
                print(e)
            update_full_text = """
            UPDATE `papers`
            SET full_text = %s
            WHERE title = %s
            """
            utils.db.execute(cursor, update_full_text, (result['full_text'], same_html_record['title']))

    else:
        # If no record exists with the same HTML URL, scrape the new record
        scraped_record: Dict[str, str] = scrape.scrape(url, compile=False)

        # Check if a record with the same title exists
        same_title_query = """
        SELECT id
        FROM `papers`
        WHERE title = %s
        """
        same_title_record = utils.db.execute(cursor, same_title_query, (scraped_record['title'],))

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
            update_urls = """
            UPDATE `papers`
            SET urls = %s
            WHERE title = %s
            """
            utils.db.execute(cursor, update_urls, (json.dumps(urls), same_title_record['title']))

            # check full-text
            if same_title_record['full_text'] is None:
                try:
                    result['full_text'] = search.extract_text(scraped_record['pdf_url'])
                except Exception as e:
                    print(e)
                update_full_text = """
                UPDATE `papers`
                SET full_text = %s
                WHERE title = %s
                """
                utils.db.execute(cursor, update_full_text, (result['full_text'], same_title_record['title']))
        else:
            # Insert the new record if no matching title is found
            html_url = scraped_record['html_url']
            pdf_url = scraped_record['pdf_url']
            del scraped_record['html_url'], scraped_record['pdf_url']
            result.update(scraped_record)
            result['urls'] = {'html': [html_url], 'pdf': [pdf_url]}
            try:
                result['full_text'] = search.extract_text(pdf_url)
            except Exception as e:
                print(e)
            insert_query = """
            INSERT INTO `papers` (
                id, code_names, title, urls, pub_name, pub_date, authors, 
                abstract, full_text, comments
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            utils.db.execute(cursor, insert_query, (
                result['id'], json.dumps(result['code_names']), result['title'],
                json.dumps(result['urls']), result['pub_name'], result['pub_date'],
                json.dumps(result['authors']), result['abstract'],
                result['full_text'], json.dumps(result['comments']),
            ))

    return scrape.utils.compile_markdown(**result)


def scrape_task(url: str, idx: int, total: int, cursor) -> Tuple[str, Union[str, Exception]]:
    """Helper function to scrape a single URL."""
    try:
        logging.info(f"[{idx+1}/{total}] Scraping {url}")
        result = scrape_or_query(url=url, cursor=cursor)
        return (url, result)  # Success
    except Exception as err:
        return (url, err)  # Failure


def main(urls: List[str], num_workers: Optional[int] = 1) -> None:
    """
    Arguments:
        urls (List[str]): A list of html urls to scrape from.
        num_workers (int): Number of threads to use for processing. Default is 1.
    """
    assert isinstance(urls, list), f"{type(urls)=}"
    assert all(isinstance(x, str) for x in urls)
    
    urls_unique: List[str] = []
    for url in urls:
        if url not in urls_unique:
            urls_unique.append(url)
    if len(urls_unique) != len(urls):
        logging.info(f"Provided list contains duplicates. Reduced to {len(urls_unique)} papers.")

    # Connect to the database
    conn, cursor = utils.db.init()

    successes: List[Tuple[str, str]] = []
    failures: List[Tuple[str, str]] = []

    # Thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        tasks = [
            executor.submit(scrape_task, url, idx, len(urls_unique), cursor)
            for idx, url in enumerate(urls_unique)
        ]

        for future in tasks:
            url, result = future.result()
            if isinstance(result, str):  # Success (result contains scraped data)
                successes.append((url, result))
            else:  # Failure (result contains error message)
                assert isinstance(result, Exception)
                failures.append((url, str(result)))

    # Commit and close the database connection
    conn.commit()
    if conn.is_connected():
        cursor.close()
        conn.close()

    # Write results to the file once
    filepath = os.path.join("scrape", "scraped_papers.md")
    with open(filepath, mode='w', encoding='utf8') as f:
        for _, result in successes:
            f.write(result)

    # Log failures to log.json
    with open("log.json", mode='w') as f:
        f.write(jsbeautifier.beautify(json.dumps(failures), jsbeautifier.default_options()))

    # Log the process termination
    logging.info("Process terminated.")


if __name__ == "__main__":
    from scrape.papers_to_scrape import papers_to_scrape
    main(urls=papers_to_scrape)
