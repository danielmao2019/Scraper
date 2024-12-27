from scrape import scrape
from scrape.papers_tmp import papers_tmp

import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


def main(url_dict):
    """
    Arguments:
        url_dict (dict): dictionary of lists of urls to be scrapped.
    """
    assert type(url_dict) == dict, f"type(url_dict)={type(url_dict)}"
    for group in url_dict:
        url_list = url_dict[group]
        logging.info(f"Processing group '{group}'." + (
            " Nothing given." if len(url_list) == 0 else f" {len(url_list)} urls found."))
        # get unique urls and preserve original order
        url_list_unique = []
        for url in url_list:
            if url not in url_list_unique:
                url_list_unique.append(url)
        if len(url_list_unique) != len(url_list):
            logging.info(f"Provided list contains duplicates. Reduced to {len(url_list_unique)} papers.")
        # iterate through unique url list and scrape each
        dst = f"scrape/scraped_papers_{group}.md"
        with open(dst, mode='w', encoding='utf8') as f:
            for idx, url in enumerate(url_list_unique):
                logging.info(f"[{idx+1}/{len(url_list_unique)}] Scraping {url}")
                f.write(scrape(url))
    logging.info(f"Process terminated.")


if __name__ == "__main__":
    main(url_dict=papers_tmp)
