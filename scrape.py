from typing import Tuple, List, Dict, Any, Union
import os
import json
import jsbeautifier
from concurrent.futures import ThreadPoolExecutor
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
import psycopg2
import scrape
import search
import utils


def scrape_or_query(url: str, cursor: psycopg2.extensions.cursor) -> str:
    result = {key: None for key in [
        'id', 'code_names', 'title', 'urls',
        'pub_name', 'pub_date', 'authors', 'abstract',
        'comments',
    ]}

    # Query to check if the HTML URL exists in the `urls->html` array
    same_html_query = """
        SELECT code_names, title, urls, pub_name, pub_date, authors, abstract, comments,
            full_text IS NULL AS missing_full_text
        FROM papers WHERE urls->'html' ? %s
    """
    same_html_record = utils.db.execute(cursor, same_html_query, (url,))

    if same_html_record:
        # Cleanup existing record
        assert len(same_html_record) == 1, f"same_html_record={json.dumps(same_html_record)}"
        same_html_record = same_html_record[0]
        missing_full_text = same_html_record['missing_full_text']
        del same_html_record['missing_full_text']

        # Update result using existing record
        assert set(same_html_record.keys()).issubset(set(result.keys()))
        result.update(same_html_record)

        # Complete full-text scraping
        if missing_full_text:
            urls = json.loads(same_html_record['urls'])
            urls = [(html, pdf) for html, pdf in zip(urls['html'], urls['pdf']) if html == url]
            assert len(urls) == 1, f"{urls=}"
            try:
                full_text = search.extract_text(urls[0][1])
            except Exception as e:
                print(e)
                full_text = None
            update_full_text = "UPDATE papers SET full_text = %s WHERE title = %s"
            utils.db.execute(cursor, update_full_text, (full_text, same_html_record['title']))

    else:
        # If no record exists with the same HTML URL, scrape the new record
        scraped_record: Dict[str, Any] = scrape.scrape(url, compile=False)

        # Check if a record with the same title exists
        same_title_query = """
            SELECT code_names, title, urls, pub_name, pub_date, authors, abstract, comments,
                full_text IS NULL AS missing_full_text
            FROM papers WHERE title = %s
        """
        same_title_record = utils.db.execute(cursor, same_title_query, (scraped_record['title'],))

        if same_title_record:
            # Cleanup existing record
            assert len(same_title_record) == 1, f"{same_title_record=}"
            same_title_record = same_title_record[0]
            missing_full_text = same_title_record['missing_full_text']
            del same_title_record['missing_full_text']
            same_title_record['urls']['html'].append(scraped_record['html_url'])
            same_title_record['urls']['pdf'].append(scraped_record['pdf_url'])

            # Update result using existing record
            assert set(same_title_record.keys()).issubset(set(result.keys()))
            result.update(same_title_record)

            # Update database with additional urls
            update_urls = "UPDATE papers SET urls = %s WHERE title = %s"
            utils.db.execute(cursor, update_urls, (json.dumps(same_title_record['urls']), same_title_record['title']))

            # Complete full-text scraping
            if missing_full_text:
                try:
                    full_text = search.extract_text(scraped_record['pdf_url'])
                except Exception as e:
                    print(e)
                    full_text = None
                update_full_text = "UPDATE papers SET full_text = %s WHERE title = %s"
                utils.db.execute(cursor, update_full_text, (full_text, same_title_record['title']))

        else:
            # Cleanup scraped record
            html_url = scraped_record['html_url']
            pdf_url = scraped_record['pdf_url']
            del scraped_record['html_url'], scraped_record['pdf_url']
            scraped_record['urls'] = {'html': [html_url], 'pdf': [pdf_url]}

            # Update result with scraped record
            assert set(scraped_record.keys()).issubset(set(result.keys()))
            result.update(scraped_record)

            # Complete full-text scraping
            try:
                full_text = search.extract_text(pdf_url)
            except Exception as e:
                print(e)
                full_text = None

            # Update database with scraped record
            insert_query = """
                INSERT INTO papers (
                    id, code_names, title, urls, pub_name, pub_date, authors, 
                    abstract, full_text, comments
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            utils.db.execute(cursor, insert_query, (
                result['id'], result['code_names'], result['title'], json.dumps(result['urls']),
                result['pub_name'], result['pub_date'], result['authors'], result['abstract'],
                full_text, result['comments'],
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


def main(urls: List[str], num_workers: int) -> None:
    """
    Arguments:
        urls (List[str]): A list of html urls to scrape from.
        num_workers (int): Number of threads to use for processing.
    """
    assert isinstance(urls, list), f"{type(urls)=}"
    assert all(isinstance(x, str) for x in urls)
    assert isinstance(num_workers, int), f"{type(num_workers)=}"

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

    # Close the database connection
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default=1)
    args = parser.parse_args()
    main(urls=papers_to_scrape, num_workers=args.num_workers)
