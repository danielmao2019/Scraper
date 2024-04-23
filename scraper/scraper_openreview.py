import json

from . import utils


def scrape_openreview(url: str) -> dict:
    assert type(url) == str, f"{type(url)=}"
    soup = utils.get_soup(url)
    # construct json
    json_str = soup.findAll('script', type="application/json")
    assert len(json_str) == 1
    json_str = json_str[0].text.strip()
    json_dict = json.loads(json_str)
    json_dict = json_dict['props']['pageProps']['forumNote']
    # extract from json
    title = utils.get_value(json_dict['content']['title'])
    pdf_url = utils.get_pdf_url(url, soup.find('a', title="Download PDF")['href'])
    pub_name, pub_year = utils.parse_writers(json_dict['writers'] if len(json_dict['writers']) else json_dict['content']['venue'])
    pub_year = f"`{pub_year}`"
    authors = ", ".join(utils.get_value(json_dict['content']['authors']))
    abstract = utils.get_value(json_dict['content']['abstract'])
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_year': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
