import json
import os
import re


def write_default_key_template():
    return 'EVENT: ID 1\nTRIGGER: {}\nAGENT: {}\nPATIENT: {}\nTIME: {}\nPLACE: {}\nAIM: {}\nOLD_VALUE: {}\nNEW_VALUE: {}\nVARIATION: {}\nCAUSE: {}\nTENSE: \nCOMMENTS: \n'


def parse_one_group(group_root, filename="total.json"):
    for fn in os.listdir(group_root):
        if len(re.findall("^[1-5].", fn)) != 0:
            os.remove(os.path.join(group_root, fn))
    with open(os.path.join(group_root, filename), "r", encoding="utf-8") as f:
        items = json.load(f)
    with open(os.path.join(group_root, "news.txt"), "w", encoding="utf-8") as f:
        for no, item in enumerate(items):
            no += 1
            f.write("## DOC %d START\n" % no)
            date_str = item["date_publish"]
            title = item["gntitle"]
            description = item["description"]
            text = item["text"]

            content = date_str + "\n" + title

            if description is not None:
                content = content + "\n\n" + description
            elif text is not None:
                content = content + "\n\n" + text.split("\n")[0]

            f.write("%s\n" % content.strip())
            f.write("## DOC %d END\n\n" % no)
    if not os.path.exists(os.path.join(group_root, "key.txt")):
        with open(os.path.join(group_root, "key.txt"), "w", encoding="utf-8") as f:
            f.write(write_default_key_template())


def parse_one_crawl(root):
    for no in os.listdir(root):
        if no.startswith("."):
            continue
        dir = os.path.join(root, no)
        parse_one_group(dir)
        print("\t\tProcessing to %s" % dir)


if __name__ == "__main__":
    root = "/Users/liuxiao/Dropbox/MyProjects/Schema/data_annotation/unlabeled_data"
    for dt in os.listdir(root):
        if dt.startswith(".") or "." in dt:
            continue
        dir = os.path.join(root, dt)
        print("Processing crawl split %s" % dir)
        parse_one_crawl(dir)
