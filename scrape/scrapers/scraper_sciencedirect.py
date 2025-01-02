from typing import Dict, Any


def scrape_sciencedirect(url: str) -> Dict[str, Any]:
    assert type(url) == str, f"type(url)={type(url)}"
    # get Elsevier client
    from elsapy.elsclient import ElsClient
    config = {
        "apikey": "1be9aac6ca066332daef54b7ff83996e",
        "insttoken": ""
    }
    client = ElsClient(config['apikey'])
    client.inst_token = config['insttoken']
    # get pii
    from elsapy.elsdoc import FullDoc
    id = url.split('/')[-1]
    pii_doc = FullDoc(sd_pii=id)
    assert pii_doc.read(client)
    data = pii_doc.__dict__['_data']['coredata']
    # get pdf url
    if url.endswith('/'):
        url = url[:-1]
    pdf_url = url + "/pdfft"
    # get title
    title = data['dc:title']
    # get authors
    authors = [
        a['$'].split(", ")[1] + ' ' + a['$'].split(", ")[0]
        for a in data['dc:creator']
    ]
    # get pub name
    pub_name = data['prism:publicationName']
    # get pub year
    pub_year = data['prism:coverDate']
    # get abstract
    abstract = data['dc:description']
    return {
        'title': title,
        'html_url': url,
        'pdf_url': pdf_url,
        'pub_name': pub_name,
        'pub_date': pub_year,
        'authors': authors,
        'abstract': abstract,
    }
