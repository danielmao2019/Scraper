import pytest
import sys
sys.path.append("../..")
from search.search_in_file import search_in_file


@pytest.mark.parametrize("url, keyword, count", [
    (
        "https://www.mdpi.com/1424-8220/18/4/966/pdf?version=1522057381",
        "graph cut",
        38,
    ),
    (
        "https://link.springer.com/content/pdf/10.1023/A:1007379606734.pdf",
        "inductive bias",
        16,
    ),
    (
        "https://www.mdpi.com/2072-4292/7/8/9682/pdf?version=1438244960",
        "octree",
        54,
    ),
    (
        "https://arxiv.org/pdf/1604.01685.pdf",
        "pascal",
        19,
    ),
    (
        "https://arxiv.org/pdf/1604.01685.pdf",
        "pascal context",
        5,
    ),
    (
        "https://arxiv.org/pdf/1604.01685.pdf",
        "pascal-context",
        5,
    ),
    (
        "https://arxiv.org/pdf/1707.08114.pdf",
        "task relation",
        52,
    ),
    (
        "https://www.mdpi.com/2072-4292/7/8/9682/pdf?version=1438244960",
        "triangulated irregular network",
        2,
    ),
    (
        "https://www.sciencedirect.com/science/article/pii/S0924271616304026/pdfft",
        "object based",
        20,
    ),
    (
        "https://journals.scholarsportal.info/details/09265805/v124icomplete/nfp_abcdwacopc.xml",
        "change detection",
        34,
    ),
    (
        "https://journals.scholarsportal.info/details/09265805/v124icomplete/nfp_abcdwacopc.xml",
        "amodal",
        8,
    ),
])
def test_search(url: str, keyword: str, count: int) -> None:
    result = search_in_file(url, [keyword])
    assert type(result) == dict and set(result.keys()) == set([keyword])
    assert result[keyword] == count
