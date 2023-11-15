"""Import the papers from papers.txt file"""

import pathlib
import sys
import urllib.request as request

CWD = pathlib.Path.cwd()
PAPERS_PATH = CWD / "papers.txt"
OUT_PAPERS_DIR = CWD / "papers"
OUT_PAPERS_DIR.mkdir(exist_ok=True)
HELP_MESSAGE = """usage: download [--force|--help]

--force: Force download of files.

--help: Show this help message.
"""


def split_link_file_pair(link_file_pair: str) -> tuple[str, str]:
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


def parse_paper_links() -> dict[str, str]:
    if not PAPERS_PATH.exists():
        print('Could not find "papers.txt" file in current working directory.')
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


def download_paper(link: str) -> bytes:
    print("Downloading paper: " + link)
    response = request.urlopen(link)
    if response.status != 200:
        print("Status code not ok for " + link)
        sys.exit(1)
    return response.read()


def parse_arguments():
    if len(sys.argv) == 1:
        return True
    if len(sys.argv) != 2:
        print("Too many arguments\n")
        print(HELP_MESSAGE)
        sys.exit(1)
    option = sys.argv[1]
    if option == "--help":
        print(HELP_MESSAGE)
        sys.exit(0)
    if option == "--force":
        return False
    print("Wrong argument\n")
    print(HELP_MESSAGE)
    sys.exit(1)


def download_papers():
    check = parse_arguments()
    paper_links = parse_paper_links()
    for file, link in paper_links.items():
        paper_path = OUT_PAPERS_DIR / file
        if check and paper_path.exists():
            print("Skipping already existing file: " + str(paper_path))
            continue
        paper_bytes = download_paper(link)
        paper_path.write_bytes(paper_bytes)
