import argparse
import os
import shutil
from tqdm import tqdm

selection = {'ita', 'fra', 'eng', 'deu', 'esp'}


def main(args):
    count = 0
    for _ in os.listdir(args.input):
        count += 1

    for celex in tqdm(os.listdir(args.input), total=count):
        for filename in os.listdir(os.path.join(args.input, celex)):
            if not filename.endswith('.html'):
                continue

            lang = filename.split('_')[1]
            if lang in selection:
                shutil.copy(os.path.join(args.input, celex, filename), os.path.join(args.output, lang))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input folder',
                        default='../input/html')
    parser.add_argument('-o', '--output', type=str, help='output folder',
                        default='../output/html')

    args = parser.parse_args()
    main(args)
