from typing import List
import re
import os
from concurrent.futures import ThreadPoolExecutor
import psycopg2
import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
import utils
from scrape.utils import compile_markdown


def _keyword2regex(keyword: str) -> str:
    result = "(?:\s+|\s*-\s*)".join([
        "-?\n?".join(list(w)) for w in keyword.split(' ')
    ])
    return result


ROOT_DIR = "search/results"


def process_keyword(filters: List[str], keyword: str, output_dir: str, cursor: psycopg2.extensions.cursor) -> None:
    """
    Processes a single keyword by querying the database and writing results to a markdown file.

    Arguments:
        filters (List[str]): List of filter keywords for the initial filtering step.
        keyword (str): The primary keyword to process.
        output_dir (str): Directory to store the output markdown file.
        cursor: Database cursor object.
    """
    logging.info(f"Processing keyword: {keyword}")

    # Build the filter query dynamically
    filter_conditions = " AND ".join(
        ["REGEXP_COUNT(full_text, %s, 1, 'i') > 0"] * len(filters)
    )

    query = f"""
        WITH filtered_papers AS (
            SELECT
                title, urls, pub_name, pub_date, authors, abstract,
                REGEXP_COUNT(full_text, %s, 1, 'i') AS match_count
            FROM papers
            WHERE {filter_conditions}
        )
        SELECT
            title, urls, pub_name, pub_date, authors, abstract,
            match_count
        FROM filtered_papers
        WHERE match_count > 0
        ORDER BY match_count DESC, (urls->'pdf'->>0) ASC;
    """

    params = [_keyword2regex(keyword)] + [_keyword2regex(f) for f in filters]
    matched_papers = utils.db.execute(cursor, query, tuple(params))

    # Write results to a markdown file
    output_filepath = os.path.join(output_dir, f"{re.sub(pattern=' ', repl='_', string=keyword)}.md")
    with open(output_filepath, mode='w', encoding='utf8') as f:
        for paper in matched_papers:
            f.write(f"count={paper['match_count']}\n")
            f.write(compile_markdown(**paper))
            f.write('\n')

    logging.info(f"Results for keyword '{keyword}' written to {output_filepath}")


def main(output_dir: str, filters: List[str], keywords: List[str], num_workers: int) -> None:
    """
    Searches for keywords in the full_text column of a database table.
    Writes the results to markdown files for each keyword using multi-threading.

    Arguments:
        output_dir (str): Directory to store the output markdown files.
        filters (List[str]): A list of filter keywords for hierarchical filtering.
        keywords (List[str]): A list of primary keywords to search in the database.
        num_workers (int): Number of worker threads to use for parallel processing.
    """
    assert isinstance(keywords, list) and all(isinstance(k, str) for k in keywords), \
        "Keywords must be a list of strings (regular expressions)."
    assert isinstance(filters, list) and all(isinstance(f, str) for f in filters), \
        "Filters must be a list of strings (regular expressions)."

    output_dir = os.path.join(ROOT_DIR, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Connect to the database
    conn, cursor = utils.db.init()

    # Use ThreadPoolExecutor for multi-threaded keyword processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_keyword, filters, keyword, output_dir, cursor)
            for keyword in keywords
        ]
        # Wait for all threads to complete
        for future in futures:
            future.result()

    cursor.close()
    conn.close()
    logging.info("Keyword search completed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--output-dir', type=str, required=True,
                        help="Directory to store the output markdown files.")
    parser.add_argument('-f', '--filters', nargs='+', default=[],
                        help="List of filters for hierarchical search.")
    parser.add_argument('-k', '--keywords', nargs='+', required=True,
                        help="List of keywords to search in the database.")
    parser.add_argument('-n', '--num-workers', type=int, default=1,
                        help="Number of worker threads for parallel processing.")
    args = parser.parse_args()
    main(output_dir=args.output_dir, filters=args.filters, keywords=args.keywords, num_workers=args.num_workers)
