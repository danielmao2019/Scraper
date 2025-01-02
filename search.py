from typing import List
import re
import os
from concurrent.futures import ThreadPoolExecutor
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


def process_keyword(keyword: str, output_dir: str, cursor) -> None:
    """
    Processes a single keyword by querying the database and writing results to a markdown file.

    Arguments:
        keyword (str): The keyword to process.
        output_dir (str): Directory to store the output markdown file.
        cursor: Database cursor object.
    """
    logging.info(f"Processing keyword: {keyword}")

    # Use SQL to find matches and count occurrences
    query = """
        WITH
        paper_matches AS (
            SELECT 
                title, urls, pub_name, pub_date, authors, abstract,
                REGEXP_COUNT(full_text, %s, 1, 'i') AS match_count
            FROM papers
        )
        SELECT 
            title, urls, pub_name, pub_date, authors, abstract,
            match_count
        FROM paper_matches
        WHERE match_count > 0
        ORDER BY match_count DESC;
    """
    matched_papers = utils.db.execute(cursor, query, (_keyword2regex(keyword),))

    # Write results to a markdown file
    output_filepath = os.path.join(output_dir, f"{re.sub(pattern=' ', repl='_', string=keyword)}.md")
    with open(output_filepath, mode='w', encoding='utf8') as f:
        for paper in matched_papers:
            f.write(f"count={paper['match_count']}\n")
            f.write(compile_markdown(**paper))
            f.write('\n')

    logging.info(f"Results for keyword '{keyword}' written to {output_filepath}")

def main(output_dir: str, keywords: List[str]) -> None:
    """
    Searches for keywords in the full_text column of a database table.
    Writes the results to markdown files for each keyword using multi-threading.

    Arguments:
        output_dir (str): Directory to store the output markdown files.
        keywords (List[str]): A list of regular expressions representing keywords.
    """
    assert isinstance(keywords, list) and all(isinstance(k, str) for k in keywords), \
        "Keywords must be a list of strings (regular expressions)."
    output_dir = os.path.join(ROOT_DIR, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Connect to the database
    conn, cursor = utils.db.init()

    # Use ThreadPoolExecutor for multi-threaded keyword processing
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_keyword, keyword, output_dir, cursor)
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
    parser.add_argument('-d', '--output-dir', type=str)
    parser.add_argument('-k', '--keywords', nargs='+', default=[])
    args = parser.parse_args()
    main(output_dir=args.output_dir, keywords=args.keywords)
