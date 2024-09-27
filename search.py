import os
import re
import tqdm
from PyPDF2 import PdfReader


def search_in_file(pdf_url: str, keyword: str) -> bool:
    """
    Args:
        context (int): the number of lines before and after the line of match.
            Default: 3.
    """
    # input checks
    assert type(keyword) == str, f"{type(keyword)=}"
    assert keyword != ""
    # download pdf to temporary file
    os.system(' '.join([
        'wget', pdf_url, '--output-document', "tmp.pdf", '--quiet',
    ]))
    # extract lines
    reader = PdfReader("tmp.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    lines = text.split('\n')
    # cleanup
    os.system(' '.join([
        'rm', "tmp.pdf",
    ]))
    # search for keyword
    return any([
        len(re.findall(pattern=keyword, string=l)) > 0
        for l in lines
    ])


def main(filepath: str, keyword: str) -> None:
    # get list of pdf urls
    with open(filepath, mode='r') as f:
        content = f.read()
    all_pdf_urls = re.findall(pattern="\((http.+\.pdf)\)", string=content)
    print(f"Found {len(all_pdf_urls)} pdf urls.")
    # search relevant documents
    failures: int = 0
    for pdf_url in tqdm.tqdm(all_pdf_urls):
        try:
            if search_in_file(pdf_url=pdf_url, keyword=keyword):
                with open(f"search_results_{keyword}.txt", mode='a') as f:
                    f.write(pdf_url + '\n')
        except Exception as e:
            failures += 1
    print(f"{failures} failed files.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', '-f', type=str)
    parser.add_argument('--keyword', '-k', type=str)
    args = parser.parse_args()
    main(filepath=args.filepath, keyword=args.keyword)
