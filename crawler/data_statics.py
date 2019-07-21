import os


def start_calculation(group_root):
    n_groups, n_documents, n_events = 0, 0, 0

    for date_name in os.listdir(group_root):
        if date_name.startswith("."):
            continue
        print("iterating over %s..." % date_name)
        for group_no in os.listdir(os.path.join(group_root, date_name)):
            if group_no.startswith("."):
                continue
            n_groups += 1

            with open(os.path.join(group_root, date_name, group_no, "news.txt"), "r", encoding="utf-8") as f:
                all_content = f.readlines()
                n_documents += len(list(filter(lambda x: x.startswith("## "), all_content))) / 2

            if os.path.exists(os.path.join(group_root, date_name, group_no, "key.txt")):
                with open(os.path.join(group_root, date_name, group_no, "key.txt"), "r", encoding="utf-8") as f:
                    all_content = f.readlines()
                    n_events += len(list(filter(lambda x: x.startswith("EVENT: ID"), all_content)))

    return n_groups, n_documents, n_events


if __name__ == "__main__":
    group_root = "/Users/liuxiao/Dropbox/MyProjects/Schema/data_annotation/unlabeled_data"
    n_groups, n_documents, n_events = start_calculation(group_root)
    print("There are %d groups, %d documents and %d events labeled." % (n_groups, n_documents, n_events))
