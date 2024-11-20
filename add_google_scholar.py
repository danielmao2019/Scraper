from typing import List
import time
import re
from scholarly import scholarly, ProxyGenerator


def get_bibtex_from_google_scholar(url: str) -> str:
    """
    Retrieve the BibTeX entry for a given article URL from Google Scholar.
    
    Parameters:
        url (str): The URL of the article (arXiv, IEEE Xplore, etc.)
    
    Returns:
        str: The BibTeX string if found, or an error message.
    """
    try:
        # Set up the ProxyGenerator to avoid rate-limiting
        pg = ProxyGenerator()
        assert pg.FreeProxies()  # Use free proxies
        scholarly.use_proxy(pg)
        
        # Search for the article by URL
        search_query = scholarly.search_pubs(url)
        article = next(search_query, None)

        if not article:
            return "No matching article found on Google Scholar."

        # Access the article's BibTeX information
        bibtex_path = article['url_scholarbib']
        if not bibtex_path:
            return "No BibTeX information found for this article."

        # Build the full URL by prepending the base URL
        bibtex_url = f"https://scholar.google.com{bibtex_path}"
# https://arxiv.org/abs/1810.08462
# https://scholar.googleusercontent.com/scholar.bib?q=info:6Y8A8lcg8gkJ:scholar.google.com/&output=citation&scisdr=ClExCJo7EKSziYtFXaE:AFWwaeYAAAAAZz1DRaFzfdqf-zb7h-p2wuR276I&scisig=AFWwaeYAAAAAZz1DRcYO2GdcNCPbyBabFiRZ7lo&scisf=4&ct=citation&cd=-1&hl=en
# https://scholar.google.com/scholar?hl=en&q=info:6Y8A8lcg8gkJ:scholar.google.com/&output=cite&scirp=0&hl=en
        return bibtex_url

    except Exception as e:
        return f"An error occurred: {str(e)}"


def process_document(filepath: str) -> None:
    with open(filepath, mode='r') as f:
        lines = f.readlines()
    result: List[str] = []
    idx = 0
    while idx < len(lines):
        if "[[abs-" in lines[idx]:
            result.append(lines[idx])
            assert idx + 1 < len(lines)
            assert "[[pdf-" in lines[idx+1]
            result.append(lines[idx+1])
            url = re.findall(pattern="\[\[abs-.*\]\((.+)\)\]", string=lines[idx])
            assert len(url) == 1, f"{lines[idx]=}, {url=}"
            url = url[0]
            print(f"Processing {url}...")
            bibtex = get_bibtex_from_google_scholar(url=url)
            result.append(f"    [[bibtex]({bibtex})]\n")
            idx = idx + 2
        else:
            result.append(lines[idx])
            idx = idx + 1
        time.sleep(5)
    result = ''.join(result)
    with open(filepath, mode='w') as f:
        f.write(result)


if __name__ == "__main__":
    filepath = "/home/d_mao/repos/Machine-Learning-Knowledge-Base/paper-collections/deep-learning/applications/2D-vision/image/change_detection.md"
    process_document(filepath)
