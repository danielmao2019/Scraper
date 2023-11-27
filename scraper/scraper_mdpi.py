import requests
from bs4 import BeautifulSoup


def scrape_mdpi(url):
    response = requests.get(url)
    assert response.status_code == 200, f"{response.status_code=}"
    soup = BeautifulSoup(response.content, "html.parser")
    # get title
    title = soup.find("h1", class_="title hypothesis_container").text.strip()
    # get year
    year = soup.find("div", class_="pubhistory").text.strip()
    year = year.split('/')[0].split(':')[1].strip().split(' ')
    year[0] = '0' * (2-len(year[0])) + year[0]
    year[1] = year[1][:3]
    year[2] = '`' + year[2] + '`'
    year = ' '.join(year)
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
