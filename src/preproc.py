import argparse
import os

import datasets
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import udf, col, first, rand

import matplotlib
import matplotlib.pyplot as plt

import json
import nltk
import kenlm
from langdetect import detect

matplotlib.use('Agg')


def plot_column(path, df_perplexity, name):
    score_list = [row[name] for row in df_perplexity.select(name).collect()]
    plt.hist(score_list, bins=100, color='blue', alpha=0.7)
    plt.ylabel('Frequency')
    plt.xlabel(name)
    plt.title(name)
    plt.grid(True)
    plt.savefig(os.path.join(path, name + '.png'))
    plt.close()


@udf(returnType=FloatType())
def sentence_perplexity(sentence):
    tokens = ' '.join(nltk.word_tokenize(sentence)).lower()
    return model.score(tokens)


@udf
def ngrams(text):
    words = text.lower().split(' ')
    return [x for x in words[:args.ngram] if x.isalpha()]


@udf(returnType=FloatType())
def alphabetic_percentage(text):
    alphabetic_count = sum(1 for char in text if char.isalpha())
    line_length = len(text.replace(" ", ""))
    percentage = (alphabetic_count / line_length) * 100
    return percentage


@udf(returnType=StringType())
def lang_filter(text):
    split = text.split(" ")
    return detect(" ".join(split[:50]))


def extract_sections(x, exclude=None, lengths=None, removed_lines=None, celex_ita=None, section_breaks=None):
    json_obj = json.loads(x[1])
    data = []

    path = x[0]
    splits = path.split('/')
    celex = splits[-1].split('.')[0]
    lang = splits[-2]
    id = 0
    len_doc = 0

    if exclude and lang == exclude:
        return []

    if celex_ita is not None and celex not in celex_ita:
        return []

    for key in json_obj:
        if key == "info":
            continue
        len_doc += len(json_obj[key]['text'])
        len_doc += len(json_obj[key]['titles'])

    section_text = []
    for key in json_obj:
        if key == "info":
            continue
        titles = json_obj[key]['titles']
        texts = json_obj[key]['text']

        if section_breaks is None:
            for title in titles:
                section_text.append(title)
                id += 1
            for text in texts:
                section_text.append(text)
                id += 1

            data.append((celex, lang, len_doc, id, '\n'.join(section_text)))
            section_text = []
        else:
            for title in titles:
                if id in section_breaks[celex]:
                    data.append((celex, lang, len_doc, id, '\n'.join(section_text)))
                    section_text = []
                    section_text.append(title)
                else:
                    section_text.append(title)
                id += 1
            for text in texts:
                if id in section_breaks[celex]:
                    data.append((celex, lang, len_doc, id, '\n'.join(section_text)))
                    section_text = []
                    section_text.append(text)
                else:
                    section_text.append(text)
                id += 1

    if lengths and len_doc != lengths[celex]:
        return []

    filtered_data = []
    if removed_lines:
        for entry in data:
            celex, lang, length_doc, id, text = entry
            if celex in removed_lines and id not in removed_lines[celex]:
                filtered_data.append(entry)
        return filtered_data

    return data


def main(args):
    schema_section = StructType([
        StructField('celex', StringType(), True),
        StructField('language', StringType(), True),
        StructField('length_doc', IntegerType(), True),
        StructField('id', IntegerType(), True),
        StructField('value', StringType(), True)
    ])

    spark = SparkSession.builder \
        .config('spark.driver.host', '127.0.0.1') \
        .config("spark.executor.memory", "40g") \
        .config("spark.driver.memory", "40g") \
        .config("spark.memory.offHeap.enabled", "true") \
        .config("spark.memory.offHeap.size", "32g") \
        .config("spark.sql.broadcastTimeout", "1000") \
        .appName("sampleCodeForReference") \
        .getOrCreate()

    rdd_ita = spark.sparkContext.wholeTextFiles(os.path.join(args.input, 'ita', '*.json'))
    rdd_ita = rdd_ita.flatMap(lambda x: extract_sections(x))
    sections_ita = rdd_ita.map(lambda x: (x[0], x[2])).collectAsMap()
    section_breaks = rdd_ita.filter(lambda x: x[3] is not None) \
        .map(lambda x: (x[0], x[3])) \
        .groupByKey() \
        .mapValues(list) \
        .collectAsMap()

    df_original_ita = rdd_ita.toDF(schema=schema_section)
    df_original_ita = df_original_ita.drop("section_break")

    # word count
    before = rdd_ita.count()
    df_ita = df_original_ita
    count_words = udf(lambda x: x.count(' ') + 1, IntegerType())
    df_ita = df_ita.withColumn("wordsCount", count_words(col('value')))
    plot_column(args.output, df_ita, "wordsCount")
    df_ita = df_ita.where(df_ita['wordsCount'] >= args.length)
    after = df_ita.count()
    print('ITA sections {0}\nAfter word count filter {1}\n'.format(before, after))

    # ngram dedup
    df_ngrams = df_ita.withColumn("ngrams", ngrams(col('value')))
    df_ita = df_ngrams.dropDuplicates(["ngrams"])
    before = df_ngrams.count()
    after = df_ita.count()
    print('ITA sections {0}\nDedup ITA sections {1}'.format(before, after))
    print("\t(duplicates {})\n".format(before - after))

    # alphabetic percentage
    df_alpha = df_ita.withColumn("alphabetic_percentage", alphabetic_percentage(col('value')))
    plot_column(args.output, df_alpha, "alphabetic_percentage")
    df_ita = df_alpha.where(df_alpha['alphabetic_percentage'] >= args.alpha)
    before = after
    after = df_ita.count()
    print('ITA sections {0}\nAfter alpha filter {1}'.format(before, after))
    print("\t(lines with alphabetic percentage < {0}: {1})\n".format(args.alpha, before - after))

    # perplexity
    df_perplexity = df_ita.withColumn('perplexity', sentence_perplexity(col('value')))
    plot_column(args.output, df_perplexity, "perplexity")
    before = after
    df_ita = df_perplexity.where(df_perplexity['perplexity'] >= args.perplexity)
    after = df_ita.count()
    print('ITA sections {0}\nAfter perplexity filter {1}'.format(before, after))
    print("\t(lines with perplexity < {0}: {1})\n".format(args.perplexity, before - after))

    # language detection
    df_language = df_ita.withColumn("language_detected", lang_filter(col('value')))
    before = after
    plot_column(args.output, df_language, "language_detected")
    df_ita = df_language.where("it" == df_language.language_detected)
    after = df_ita.count()
    print('ITA sections {0}\nAfter language filter {1}'.format(before, after))
    print("\t(other than italian {})\n".format(before - after))

    df_ita = df_ita.select(['celex', 'language', 'length_doc', 'id', 'value'])
    removed_lines = df_original_ita.subtract(df_ita).select(["celex", "id"]).rdd \
        .map(lambda x: (x[0], x[1])) \
        .groupByKey() \
        .mapValues(set) \
        .collectAsMap()

    celex_ita = df_ita.select('celex').rdd.flatMap(lambda x: x).collect()

    # tutte le lingue
    rdd_dataset = spark.sparkContext.wholeTextFiles(os.path.join(args.input, '*/*.json'))
    rdd_dataset = rdd_dataset.flatMap(
        lambda x: extract_sections(x, exclude='ita', lengths=sections_ita, removed_lines=removed_lines,
                                   celex_ita=celex_ita, section_breaks=section_breaks))

    df_dataset = rdd_dataset.toDF(schema=schema_section).drop('section_break')
    df_dataset = df_dataset.union(df_ita)

    df_pivot = df_dataset.groupBy("celex", "id").pivot("language").agg(first("value"))
    df_pivot = df_pivot.orderBy(rand(seed=42))
    # df_pivot.show(100)

    # salvataggio
    print('\nWriting output')
    df_pandas = df_pivot.toPandas()
    hf_dataset = datasets.Dataset.from_pandas(df_pandas)
    hf_dataset.save_to_disk(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='input folder',
                        default='../output/json')
    parser.add_argument('-o', '--output', type=str, help='output folder',
                        default='../output')
    parser.add_argument('-b', '--binary', type=str, help='LM binary file',
                        default='../input/summaries.binary')
    parser.add_argument('-a', '--alpha', type=float, help='alphabetic percentage min',
                        default=90)
    parser.add_argument('-p', '--perplexity', type=float, help='perplexity min value',
                        default=-2000)
    parser.add_argument('-l', '--length', type=int, help='max line length',
                        default=20)
    parser.add_argument('-n', '--ngram', type=int, help='ngrams size',
                        default=10)

    args = parser.parse_args()

    model = kenlm.LanguageModel(args.binary)
    main(args)
