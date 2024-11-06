import argparse

import pandas as pd
import json
import re
import os


def main():
    prompt_types = ["queries", "titles"]
    ds = pd.read_json(args.input + ".jsonl", lines=True)
    df = pd.DataFrame(columns=[])
    for _, x in ds.iterrows():
        dic = {'celex': x['celex'], 'paragraph': x['paragraph'], 'text': x['text']}
        for prompt in prompt_types:
            pattern = r'\{(.*?)\}'

            match = re.findall(pattern, x[prompt].replace("\n", ""))

            separator = ","
            if match:
                separator = separator.join(match)
                try:
                    d = json.loads("{" + separator.replace("  ", "") + "}")
                except json.decoder.JSONDecodeError:
                    break
                try:
                    response = d[prompt]
                    dic[prompt] = response
                except KeyError:
                    break

        if dic and len(dic) == 5:
            df = df.append(dic, ignore_index=True)
    if not df.empty:
        df.to_json(args.output + '.jsonl', orient='records', lines=True)


def split_json():
    os.makedirs(args.output_split_folder, exist_ok=True)

    train_file_path = os.path.join(args.output_split_folder, "output_train.jsonl")
    test_file_path = os.path.join(args.output_split_folder, "output_test.jsonl")
    dev_file_path = os.path.join(args.output_split_folder, "output_dev.jsonl")

    with open(args.output + ".jsonl", 'r') as input_file, \
            open(train_file_path, 'w') as train_file, \
            open(test_file_path, 'w') as test_file, \
            open(dev_file_path, 'w') as dev_file:
        for line in input_file:
            data = json.loads(line)

            if data["queries"] is not None and len(data["queries"]) >= 5:
                train_queries = data["queries"][:3]
                test_queries = data["queries"][3:4]
                valid_queries = data["queries"][4:5]
            else:
                continue
            if data["titles"] is not None and len(data["titles"]) >= 5:
                train_titles = data["titles"][:3]
                test_titles = data["titles"][3:4]
                valid_titles = data["titles"][4:5]
            else:
                continue

            train_data = {"queries": train_queries, "titles": train_titles}
            test_data = {"queries": test_queries, "titles": test_titles}
            valid_data = {"queries": valid_queries, "titles": valid_titles}
            train_file.write(json.dumps(data | train_data) + "\n")
            test_file.write(json.dumps(data | test_data) + "\n")
            dev_file.write(json.dumps(data | valid_data) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='generated')
    parser.add_argument('-o', '--output', type=str, default='output_complete')
    parser.add_argument('-f', '--output_split_folder', type=str, default='split')

    args = parser.parse_args()
    main()
    split_json()
