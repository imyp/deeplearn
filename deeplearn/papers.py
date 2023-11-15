"""Import the papers from papers.txt file"""

import pathlib
import sys
import urllib.request as request

CWD = pathlib.Path.cwd()
PAPERS_PATH = CWD / "papers.txt"
OUT_PAPERS_DIR = CWD / "papers"
OUT_PAPERS_DIR.mkdir(exist_ok=True)

def split_link_file_pair(link_file_pair: str)->tuple[str, str]:
    items = link_file_pair.split(",")
    if len(items) != 2:
        print("Could not split the following line in file: \n" + link_file_pair)
        sys.exit(1)
    file = items[0]
    if not file.endswith(".pdf"):
        print(file + " is not a pdf file")
        sys.exit(1)
    link = items[1]
    if not link.startswith("http"):
        print(link + " is not a valid link")
    return file, link



def parse_paper_links()-> dict[str, str]:
    if not PAPERS_PATH.exists():
        print("Could not find \"papers.txt\" file in current working directory.")
        sys.exit(1)
    paper_collection: dict[str, str] = {}
    link_file_pairs = PAPERS_PATH.read_text().splitlines()
    for link_file_pair in link_file_pairs:
        file, link = split_link_file_pair(link_file_pair)
        if file in paper_collection:
            print("The file " + file + " is duplicated. Stopping processing.")
            sys.exit(1)
        paper_collection[file] = link
    return paper_collection
        


def save_paper(file: str, link: str):
    paper_path = OUT_PAPERS_DIR / file
    print("Downloading paper: " + link)
    response = request.urlopen(link)
    if response.status != 200:
        print("Status code not ok for " + link)
        sys.exit(1)
    print("saving to " + str(paper_path))
    paper_path.write_bytes(response.read())


def download_papers():
    paper_links = parse_paper_links()
    for file, link in paper_links.items():
        save_paper(file, link)
