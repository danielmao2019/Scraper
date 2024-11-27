from typing import List, Dict
import re
import utils


def scrape_acm(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    # get pdf url
    pdf_url = soup.findAll(name='a', attrs={'title': "View PDF"})
    pdf_url = pdf_url[0]['href']
    # get title
    title = soup.findAll(name='meta', attrs={'property': "og:title"})
    assert len(title) == 1
    title = title[0]['content']
    title = re.findall(pattern=r"(.+): ACM Transactions on Graphics.+", string=title)
    assert len(title) == 1
    title = title[0]
    # get year
    year = soup.findAll(name='span', attrs={'class': "core-date-published"})
    assert len(year) == 1
    year = year[0].text
    # get authors
    authors_list: List[str] = [
        first_name.get_text(strip=True) + ' ' + last_name.get_text(strip=True)
        for first_name, last_name in zip(
            soup.findAll(name='span', attrs={'property': 'givenName'}),
            soup.findAll(name='span', attrs={'property': 'familyName'})
        )
    ]
    authors: List[str] = []
    for author in authors_list:
        if author not in authors:
            authors.append(author)
    authors: str = ', '.join(authors)
    # get abstract
    abstract = utils.parse_abstract_after_h2(soup)
    # return
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': "ACM",
        'pub_year': year,
        'authors': authors,
        'abstract': abstract,
    }
