import argparse
import json
import os
from tqdm import tqdm
from multiprocessing.pool import Pool
from unstructured.partition.html import partition_html
from collections import defaultdict


def extract_narrative(args, file_list, islast=False):
    pbar = None
    if islast:
        pbar = tqdm(total=len(file_list))

    for file_path in file_list:
        filename = os.path.basename(file_path)

        html_elements = partition_html(filename=file_path)
        element_dict = [el.to_dict() for el in html_elements]
        dict_elements = defaultdict(dict)

        current_section = ""
        titles = []
        for x in element_dict:
            elemnt_dict = {
                "titles": [],
                "text": []
            }

            if x["type"] != "Title":
                if current_section == "":
                    current_section = x
                    dict_elements[current_section['element_id']] = elemnt_dict
                    elemnt_dict["titles"] = titles
                    titles = []
                    dict_elements["info"] = {"celex": filename.split('_')[0]}

                dict_elements[current_section['element_id']]["text"].append(' '.join(x["text"].split()))
            else:
                current_section = ""
                titles.append(x['text'])

        lang_folder = os.path.join(filename.split('_')[-2])
        path = os.path.join(args.output, lang_folder)

        os.makedirs(path, exist_ok=True)
        path_file = os.path.join(path, filename.split('_')[0] + ".json")

        with open(path_file, "w", encoding='utf-8') as json_file:
            json.dump(dict_elements, json_file)

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()


def main():
    file_list = []
    for root, _, files in tqdm(os.walk(args.input)):
        for file in files:
            file_list.append(os.path.join(root, file))

    pool = Pool(args.processes)
    processes = []
    proc_splits = [file_list[i * len(file_list) // args.processes: (i + 1) * len(file_list) // args.processes]
                   for i in range(args.processes)]

    for i, split in enumerate(proc_splits):
        processes.append(pool.apply_async(extract_narrative, args=(args, split, i == args.processes - 1)))

    pool.close()
    pool.join()

    for t in processes:
        t.get()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input folder',
                        default='../output/html')
    parser.add_argument('-p', '--processes', type=int, help='num processes',
                        default=10)
    parser.add_argument('-o', '--output', type=str, help='output folder',
                        default='../output/json')
    args = parser.parse_args()
    main()
