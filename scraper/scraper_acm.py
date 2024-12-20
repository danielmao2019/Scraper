from typing import List, Dict
import utils


def scrape_acm(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.soup.get_soup(url)
    # get pdf url
    pdf_url = soup.findAll(name='a', attrs={'title': "View PDF"})
    pdf_url = pdf_url[0]['href']
    # get title
    title = soup.findAll(name='div', attrs={'id': "organizational-chart__title"})
    assert len(title) == 1
    title = title[0].text.strip()
    # get pub name
    try:
        title_and_pub = soup.findAll(name='meta', attrs={'property': "og:title"})
        assert len(title_and_pub) == 1
        title_and_pub = title_and_pub[0]['content'].split(" | ")
        assert len(title_and_pub) == 2
        pub_name, _ = utils.parse_publisher(value=title_and_pub[1])
        # sanity check
        site_name = soup.findAll(name='meta', attrs={'property': "og:site_name"})
        assert len(site_name) == 1
        site_name = site_name[0]['content']
        assert site_name == "ACM Conferences"
    except:
        pub_name = soup.findAll(name='meta', attrs={'property': "og:site_name"})
        assert len(pub_name) == 1
        pub_name = pub_name[0]['content']
    # get pub year
    pub_year = soup.findAll(name='span', attrs={'class': "core-date-published"})
    assert len(pub_year) == 1
    pub_year = pub_year[0].text.strip()
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
    abstract = utils.soup.extract_abstract(soup)
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
