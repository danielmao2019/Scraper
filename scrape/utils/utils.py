from urllib.parse import urljoin
import requests


def get_value(item):
    if type(item) == dict:
        return item['value']
    else:
        assert type(item) in [str, list]
        return item


def get_pdf_url(
    abs_url: str = None,
    rel_pdf_url: str = None,
) -> str:
    pdf_url = urljoin(base=abs_url, url=rel_pdf_url)
    # check if new url exists
    r = requests.get(pdf_url)
    assert r.status_code == 200, f"r.status_code={r.status_code}, abs_url={abs_url}, rel_pdf_url={rel_pdf_url}"
    return pdf_url
