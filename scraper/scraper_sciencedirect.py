from typing import List, Dict
import utils


def scrape_sciencedirect(url: str) -> Dict[str, str]:
    assert type(url) == str, f"type(url)={type(url)}"
    soup = utils.get_soup(url)
    title = soup.findAll(name='meta', attrs={'name': 'citation_title'})
    assert len(title) == 1
    title = title[0]['content']
    pdf_url = url + "/pdfft"
    pub_name = soup.findAll(name='meta', attrs={'name': 'citation_journal_title'})
    assert len(pub_name) == 1
    pub_name = pub_name[0]['content']
    pub_year = soup.findAll(name='meta', attrs={'name': 'citation_publication_date'})
    assert len(pub_year) == 1
    pub_year = pub_year[0]['content']
    assert len(pub_year.split('/')) == 3, f"{pub_year=}"
    pub_year = f"`{pub_year.split('/')[0]}`"
    authors = ', '.join([
        first_name.get_text(strip=True) + ' ' + last_name.get_text(strip=True)
        for first_name, last_name in zip(
            soup.findAll(name='span', attrs={'class': 'given-name'}),
            soup.findAll(name='span', attrs={'class': 'text surname'})
        )
    ])
    h2_tag = soup.find('h2', string="Abstract")
    assert h2_tag
    abstract_div = h2_tag.find_next('div')
    assert abstract_div
    def extract_text_with_spaces(element):
        text: List[str] = []
        for item in element.descendants:
            if item.name is None:
                text.append(item)
        return ''.join(text)

    abstract = extract_text_with_spaces(abstract_div)
    return {
        'title': title,
        'abs_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_year': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
