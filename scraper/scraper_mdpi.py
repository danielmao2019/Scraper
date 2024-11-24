from typing import Dict
import utils


def scrape_mdpi(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # get title
    title = soup.find("h1", class_="title hypothesis_container").text.strip()
    # get year
    year = soup.find("div", class_="pubhistory").text.strip()
    year = year.split('/')[0].split(':')[1].strip()
    # get authors
    authors = soup.find_all("a", class_="sciprofiles-link__link")
    authors = ', '.join([author.text for author in authors])
    # get abstract
    abstract = soup.find("div", class_="html-p").text.strip()
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': "",
        'pub_name': "MDPI",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
