import json
import os
import sys
from functools import partial

import yaml
from stanfordcorenlp import StanfordCoreNLP


def load_wn_words(fps=None):
    if fps is None:
        fps = ["./event", "./act"]
    x = set()
    for fp in fps:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip().split("_")[-1].lower()
                x.add(line)
    return x


wn_words = load_wn_words()


def input_queue(INPUT_DIR):
    for DATE_STR in os.listdir(INPUT_DIR):
        if DATE_STR.startswith("."): continue
        for ID in os.listdir(os.path.join(INPUT_DIR, DATE_STR)):
            if ID.startswith("."): continue
            file_path = os.path.join(INPUT_DIR, DATE_STR, ID, "news.txt")
            if os.path.exists(file_path):
                yield file_path


def process_it(ANNOTATOR, FILE_PATH, TEXT_MAXLEN):
    texts = []
    name = "-".join(FILE_PATH.split(os.sep)[-3:-1])
    flag = False
    doc_number = 0
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("##"):
                flag = True
                doc_number += 1
                continue
            if flag:  # skip all timestamps
                flag = False
                continue
            if len(line) <= TEXT_MAXLEN:
                texts.append(line.replace("|", " "))

    article = json.loads(ANNOTATOR(text="\n".join(texts)))
    return article, name, texts, doc_number / 2


def bad_entity(text):
    if text == "this":
        return True
    if text == "that":
        return True
    if text == "there":
        return True
    if text == "here":
        return True
    if text == "|":
        return True
    if text == "less":
        return True
    if text == "more":
        return True
    return False


def fix_stanford_coref(stanford_json):
    true_corefs = {}
    # get a chain
    for key, coref in stanford_json["corefs"].items():
        true_coref = []
        # get an entity mention
        for entity in coref:
            sent_num = entity["sentNum"] - 1  # starting from 0
            start_index = entity["startIndex"] - 1  # starting from 0
            end_index = entity["endIndex"] - 1  # starting from 0
            head_index = entity["headIndex"] - 1  # starting from 0
            entity_label = stanford_json["sentences"][sent_num]["tokens"][head_index]["ner"]
            entity["sentNum"] = sent_num
            entity["startIndex"] = start_index
            entity["endIndex"] = end_index
            entity["headIndex"] = head_index
            entity["headWord"] = entity["text"].split(" ")[head_index - start_index]
            entity["entityType"] = entity_label
            true_coref.append(entity)
        # check link is not empty
        if len(true_coref) > 0:
            no_representative = True
            has_representative = False
            for idx, entity in enumerate(true_coref):
                if entity["isRepresentativeMention"]:
                    if not (entity["type"] == "PRONOMINAL" or
                            bad_entity(entity["text"].lower()) or
                            len(entity["text"].split(" ")) > 5):
                        no_representative = False
                        has_representative = True
                    # remove bad representative assignments
                    else:
                        true_coref[idx]["isRepresentativeMention"] = False
            # check there exists one representative mention
            if no_representative:
                for idx, entity in enumerate(true_coref):
                    if not (entity["type"] == "PRONOMINAL" or
                            bad_entity(entity["text"].lower()) or
                            len(entity["text"].split(" ")) > 5):
                        true_coref[idx]["isRepresentativeMention"] = True
                        has_representative = True
            if has_representative:
                true_corefs[key] = true_coref
    return true_corefs


def expected_dep_type(dep_t):
    if dep_t in ["nsubjpass", "dobj", "agent", "nsubj", "poss", "tmod"]:
        return True
    for tt in ["prep_", "conj_"]:
        if dep_t.startswith(tt):
            return True
    return False


def parse_dep_edges(dep_edges, sz):
    fathers = [[] for _ in range(sz)]
    for edge in dep_edges:
        if not expected_dep_type(edge["dep"]):
            continue
        fa = edge["governor"] - 1
        u = edge["dependent"] - 1
        fathers[u].append(fa)
    return fathers


def pin_predicates(stanford_json):
    graphs = {}
    for sid, sentence in enumerate(stanford_json["sentences"]):
        dep_edges = sentence["enhancedPlusPlusDependencies"]
        fathers = parse_dep_edges(dep_edges, len(sentence["tokens"]))
        graphs[sid] = fathers

    for coref in stanford_json["corefs"].values():
        for entity in coref:
            sent_num = entity["sentNum"]
            head_index = entity["headIndex"]
            u = head_index
            fas = graphs[sent_num][u]
            entity["predicates"] = []
            for fa in fas:
                if stanford_json["sentences"][sent_num]["tokens"][fa]["pos"].startswith("VB") or \
                        (stanford_json["sentences"][sent_num]["tokens"][fa]["pos"].startswith("NN") and
                         stanford_json["sentences"][sent_num]["tokens"][fa]["lemma"].lower() in
                         wn_words):
                    predicate_lemma = stanford_json["sentences"][sent_num]["tokens"][fa]["lemma"]
                    predicate_original = stanford_json["sentences"][sent_num]["tokens"][fa]["originalText"]
                    predicate_pos = stanford_json["sentences"][sent_num]["tokens"][fa]["pos"]
                    entity["predicates"].append({
                        "lemma": predicate_lemma,
                        "original": predicate_original,
                        "pos": predicate_pos,
                        "index": fa,
                    })


def store_it(FILE, OUTPUT_PATH):
    FILE["corefs"] = fix_stanford_coref(FILE)
    pin_predicates(FILE)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(FILE, f, ensure_ascii=False)
    print("STORED %s" % OUTPUT_PATH)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("PYTHON odee_preprocess.py INPUT_DIR OUTPUT_DIR")
        sys.exit(-1)
    INPUT_DIR, OUTPUT_DIR = sys.argv[1], sys.argv[2]
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)
    with open("setting.yaml", "r") as stream:
        all_setting = yaml.load(stream)
    CORENLP_HOME = all_setting["CORENLP_HOME"]["server"]
    try:
        CORENLP_HOME.startswith("/")
        os.listdir(CORENLP_HOME)
    except:
        print("Error with CORENLP_HOME setting")
        sys.exit(-1)
    print(INPUT_DIR, OUTPUT_DIR, CORENLP_HOME)

    nlp_server = StanfordCoreNLP(CORENLP_HOME, memory='8g')
    annotator = partial(nlp_server.annotate, properties=all_setting["props"])

    cluster_count = 0
    report_count = 0
    sentence_count = 0
    token_count = 0
    for file_path in input_queue(INPUT_DIR):
        print("Processing %s" % file_path)
        file, name, content, rc = process_it(annotator, file_path, all_setting["TEXT_MAXLEN"])
        # counting somethings
        cluster_count += 1
        report_count += rc
        sentence_count += len(file["sentences"])
        for json_sentence in file["sentences"]:
            token_count += len(json_sentence["tokens"])
        # dump cleaned stanford json
        json_output_path = os.path.join(OUTPUT_DIR, name + ".json")
        store_it(file, json_output_path)
        # dump raw texts
        raw_output_path = os.path.join(OUTPUT_DIR, name + ".txt")
        with open(raw_output_path, "w", encoding="utf-8") as f:
            for line in content:
                f.write(line.strip() + "\n")

    nlp_server.close()
    print("Done for %s!" % INPUT_DIR)
    print("Total cluster count is %d." % cluster_count)
    print("Total report count is %d." % report_count)
    print("Total sentence count is %d." % sentence_count)
    print("Total token count is %d." % token_count)
