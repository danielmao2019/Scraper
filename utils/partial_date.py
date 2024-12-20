from typing import List
from dateutil.parser import parser as Parser


def parse_date(string: str) -> str:
    assert type(string) == str
    parser = Parser()
    result = parser._parse(string)[0]
    date: List[str] = []
    if result.year is not None:
        date.append(str(result.year))
        if result.month is not None:
            date.append(f"{result.month:02d}")
            if result.day is not None:
                date.append(f"{result.day:02d}")
    assert len(date) >= 1
    return '-'.join(date)


def date_eq(d1: str, d2: str) -> bool:
    d1 = d1.split('-')
    d2 = d2.split('-')
    d1 += [None] * (3-len(d1))
    d2 += [None] * (3-len(d2))
    assert len(d1) == len(d2) == 3, f"{d1=}, {d2=}"
    return str(d1) == str(d2)
