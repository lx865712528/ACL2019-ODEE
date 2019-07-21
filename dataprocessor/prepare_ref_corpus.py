import json
import os
import sys
from functools import partial

import yaml
from stanfordcorenlp import StanfordCoreNLP


def input_queue(INPUT_DIR):
    for DATE_STR in os.listdir(INPUT_DIR):
        if DATE_STR.startswith("."): continue
        for ID in os.listdir(os.path.join(INPUT_DIR, DATE_STR)):
            if ID.startswith("."): continue
            file_path = os.path.join(INPUT_DIR, DATE_STR, ID, "total.json")
            if os.path.exists(file_path):
                yield file_path


def process_it(ANNOTATOR, FILE_PATH):
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        ngs = json.load(f)

    ps = []
    sentence_count = 0
    token_count = 0
    for item in ngs:
        title = item["gntitle"]
        description = item["description"]
        text = item["text"]

        if title is None:
            title = ""
        if text is None:
            text = ""
        if description is None or description in text:
            description = ""

        content = "\n".join([title, description, text])
        content = content.replace("|", " ")
        json_article = json.loads(ANNOTATOR(text=content))
        words = []
        sentence_count += len(json_article["sentences"])
        for json_sentence in json_article["sentences"]:
            token_count += len(json_sentence["tokens"])
            for json_token in json_sentence["tokens"]:
                word = json_token["word"]
                words.append(word)
        ps.append(" ".join(words))
    return ps, sentence_count, token_count


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("PYTHON prepare_ref_corpus.py INPUT_DIR OUTPUT_PATH")
        sys.exit(-1)
    INPUT_DIR, OUTPUT_PATH = sys.argv[1], sys.argv[2]
    with open("setting.yaml", "r") as stream:
        all_setting = yaml.load(stream)
    CORENLP_HOME = all_setting["CORENLP_HOME"]["server"]
    try:
        CORENLP_HOME.startswith("/")
        os.listdir(CORENLP_HOME)
    except:
        print("Error with CORENLP_HOME setting")
        sys.exit(-1)
    print(INPUT_DIR, OUTPUT_PATH, CORENLP_HOME)

    nlp_server = StanfordCoreNLP(CORENLP_HOME, memory='8g')
    annotator = partial(nlp_server.annotate, properties=all_setting["ref_corpus_props"])

    sentence_count = 0
    token_count = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for file_path in input_queue(INPUT_DIR):
            print("Processing %s" % file_path)
            ps, sc, tc = process_it(annotator, file_path)
            sentence_count += sc
            token_count += tc
            for p in ps:
                f.write("%s\n" % p)

    nlp_server.close()
    print("Done!")
    print("Total sentence count is %d." % sentence_count)
    print("Total token count is %d." % token_count)
