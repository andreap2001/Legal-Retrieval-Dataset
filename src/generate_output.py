import argparse

import datasets
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from datasets import Dataset


def input_form_dataset():
    ds = Dataset.load_from_disk(args.input)

    non_null_ds = ds.filter(lambda example: all(value is not None for value in example.values()))
    result = []
    for x in non_null_ds:
        result.append({'celex': x['celex'], 'paragraph': x['id'], 'text': x['eng'], 'queries': "", 'titles': ""})
    return result


def main():
    torch.manual_seed(42)
    set_seed(42)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    if args.quantize:
        model = AutoModelForCausalLM.from_pretrained(args.model, token=args.auth, load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, token=args.auth)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.auth,
                                              padding_side="left")
    model.eval()
    denouncements_paragraph = input_form_dataset()
    df = pd.DataFrame()

    for document in denouncements_paragraph:
        print(document)

        prompts = [f"""
        You are a lawyer asked to create detailed and precise search engine queries based on a specific paragraph from a document. The goal is to facilitate further research and exploration related to legal, factual, or contextual aspects mentioned in the document.

        Document Paragraph:
        {document['text']}

        Instructions:
        Generate:
        Carefully read the provided document paragraph.
        Identify key topics, terms, and legal issues mentioned or implied in the paragraph.
        Create relevant search engine queries focusing on these aspects to provide comprehensive and specific information for further research.
        Ensure that the queries are detailed enough to retrieve useful and specific information related to legal precedents, case studies, laws, or regulations.
        Generate the queries in a JSON-like format with the key "queries" and the values being the generated queries.
        """,

                   f"""
         You are a lawyer asked to create titles based on a paragraph from a document.

        Document Paragraph:
        {document['text']}

        Instructions:
        Generate:
        Carefully read the provided document paragraph.
        Create titles focusing on these aspects to provide information for further research.
        Generate the queries in a JSON-like format with the key "titles" and the values being the generated titles.
         """

                   ]
        p = 0
        for x in prompts:
            with torch.no_grad():
                if (len(document['text']) > args.maxlen):
                    continue
                tokens = tokenizer(x, return_tensors="pt")
                if not args.quantize:
                    tokens = tokens.to(device)

                generated_ids = model.generate(**tokens, max_new_tokens=args.maxlen,
                                               pad_token_id=tokenizer.eos_token_id,
                                               do_sample=True, temperature=0.3, top_k=0, top_p=0.92)

                output = tokenizer.batch_decode(generated_ids[:, tokens['input_ids'].shape[1]:],
                                                skip_special_tokens=True)[0]

                if p == 0:
                    document["queries"] = output
                elif p == 1:
                    document["titles"] = output
                p += 1
        df = df.append(document, ignore_index=True)
    df.to_json(args.output+'.jsonl', orient='records', lines=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='sprarkoutput')
    parser.add_argument('-p', '--promot', type=int, default=0)
    parser.add_argument('-l', '--maxlen', type=int, default=2048)
    parser.add_argument('-m', '--model', type=str, default='mistralai/Mistral-7B-Instruct-v0.2')
    parser.add_argument('-q', '--quantize', action='store_true', default=False)
    parser.add_argument('-a', '--auth', type=str)
    parser.add_argument('-o', '--output', type=str, default='generated')

    args = parser.parse_args()
    main()
    # mistralai/Mistral-7B-Instruct-v0.2
    # mistralai/Mistral-7B-v0.1
    # meta-llama/Llama-2-7b-hf
    # meta-llama/Llama-2-13b-hf
    # meta-llama/Llama-2-70b-hf
    # meta-llama/Llama-2-13b-chat-hf
