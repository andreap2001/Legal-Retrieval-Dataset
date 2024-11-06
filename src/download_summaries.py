import argparse
import os

import pandas as pd
from tqdm import tqdm
import requests


def main(args):
    df = pd.read_csv(args.input, encoding='utf-8')
    df = df.dropna(subset=['Summary code', 'Summary production reference'])
    for ref in tqdm(df['Summary production reference']):
        try:
            url = 'https://eur-lex.europa.eu/legal-content/IT/TXT/?uri=LEGISSUM:' + ref
            page = requests.get(url, timeout=30).text
            with open(os.path.join(args.output, ref + '.html'), 'w', encoding='utf-8') as out:
                out.write(page)
        except Exception as e:
            print('Skip {0} - {1}'.format(ref, str(e)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input file',
                        default='../input/Search results 20240404.csv')
    parser.add_argument('-o', '--output', type=str, help='output folder',
                        default='../output/summaries/html')

    args = parser.parse_args()
    main(args)
