import argparse
import os

import nltk
from tqdm import tqdm
from bs4 import BeautifulSoup
import lxml.html
import lxml.html.clean
import regex


def main(args):
    cleaner = lxml.html.clean.Cleaner(style=True)

    files = [x for x in os.listdir(args.input) if x.endswith('.html')]

    for filename in tqdm(files):
        filepath = os.path.join(args.input, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            page = BeautifulSoup(file.read(), 'html.parser')
            text = page.select_one('div#text').text
            lines = text.split('\n')

            titles = [x for x in lines if x.isupper()]
            if len(titles) < 2:
                continue

            for line in lines:
                line = line.strip()
                if line == titles[-2]:
                    break

                if len(line) < 2 or line.isupper() or line[0] == '-':
                    continue

                line = line.replace(' ', ' ')
                doc = cleaner.clean_html(lxml.html.fromstring(line))
                line = doc.text_content()
                line = regex.sub(r'\n|[\s]{2,}', ' ', line)

                for sentence in nltk.sent_tokenize(line):
                    tokens = ' '.join(nltk.word_tokenize(sentence)).lower()
                    out.write(tokens + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input folder',
                        default='../output/summaries/html')
    parser.add_argument('-o', '--output', type=str, help='output folder',
                        default='../output/summaries')

    args = parser.parse_args()

    with open(os.path.join(args.output, 'summaries.txt'), 'w', encoding='utf-8') as out:
        main(args)
