import json
import os
import pickle
import random
import sys

import torch


# serve as corpus parser
class ParsedCorpus:
    def __init__(self, base_dirs: list):
        self.base_dirs = base_dirs
        self.file_names = [os.path.join(base_dir, file_name)
                           for base_dir in self.base_dirs
                           for file_name in os.listdir(base_dir)
                           if file_name.endswith(".json") and not file_name.startswith(".")]

    def __len__(self):
        return len(self.file_names)

    def get_single(self, key: str):
        try:
            assert key in ["sentences", "corefs", "pt", "answer", "file_name"]
        except:
            print("key should be 'sentences', 'corefs', 'file_name', 'pt' or 'answer'")
            sys.exit(-1)

        for file_name in self.file_names:
            # print("processing", file_name, "as", key)
            if key == "pt":
                all = torch.load(file_name + ".pt")
                yield all["fs"], all["rs"], file_name
            elif key == "answer":
                with open(file_name + ".answer", "rb") as f:
                    all = pickle.load(f)
                    yield all, file_name
            elif key == "file_name":
                yield file_name
            else:
                with open(file_name, "r", encoding="utf-8") as f:
                    news_group = json.load(f)
                    yield news_group[key], file_name

    def shuffle(self):
        random.shuffle(self.file_names)


# serve as slot realization head word vocabulary
class HeadWordVocabulary:
    def __init__(self, max_vocab: int = -1):
        self.UNK = "<UNK>"
        self.PADDING = "<PADDING>"
        self.stoi = {self.UNK: 0, self.PADDING: 1}
        self.itos = [self.UNK, self.PADDING]
        self.max_vocab = max_vocab

        # temporary
        self.counts = {}

    def __len__(self):
        return len(self.itos)

    def save(self, path="./voc.txt"):
        with open(path, "w", encoding="utf-8") as f:
            for x in self.itos:
                f.write("%s\n" % x)
        print("Save vocabulary in", path)

    def load(self, path="./voc.txt"):
        self.stoi = {}
        self.itos = []
        with open(path, "r", encoding="utf-8") as f:
            count = 0
            for line in f:
                line = line.strip()
                if len(line) == 0: continue
                self.itos.append(line)
                self.stoi[line] = count
                count += 1
        self.UNK, self.PADDING = self.itos[0:2]
        print("Load vocabulary from", path)

    def update_token(self, token: str):
        if token not in self.counts:
            self.counts[token] = 1
        else:
            self.counts[token] += 1

    def update_sentence(self, sentence: list):
        for token in sentence:
            self.update_token(token)

    def finalize_vocab(self, min_keep: int = 1, max_keep: int = 99999999):
        for k, v in sorted(self.counts.items(), key=lambda x: -x[1]):
            if min_keep <= v <= max_keep:
                index = len(self.itos)
                self.stoi[k] = index
                self.itos.append(k)
        del self.counts

    def make_vocabulary(self, corpus: ParsedCorpus, key: str = "headWord"):
        for coref, _ in corpus.get_single("corefs"):
            for realizations in coref.values():
                heads = []
                for realization in realizations:
                    if not realization["isRepresentativeMention"]:
                        continue
                    head = realization[key]
                    heads.append(head)
                self.update_sentence(heads)
        self.finalize_vocab()


class DataIterator:
    def __init__(self, corpus: ParsedCorpus, vocab: HeadWordVocabulary, entity_vocab: HeadWordVocabulary):
        self.corpus = corpus
        self.vocab = vocab
        self.entity_vocab = entity_vocab
        self.cnt = 0
        self.sentences_generator = None
        self.corefs_generator = None
        self.feature_grnerator = None
        self.reset()

    def __len__(self):
        return len(self.corpus)

    def reset(self):
        self.corpus.shuffle()
        self.cnt = 0
        del self.sentences_generator
        del self.corefs_generator
        del self.feature_grnerator
        self.sentences_generator = self.corpus.get_single("sentences")
        self.corefs_generator = self.corpus.get_single("corefs")
        self.feature_grnerator = self.corpus.get_single("pt")

    def get_minibatch(self, batch_size):
        # start_time = time.time()
        if batch_size > len(self) - self.cnt:
            batch_size = len(self) - self.cnt
        self.cnt += batch_size

        minibatch_hs = []
        minibatch_fs = []
        minibatch_rs = []
        minibatch_es = []
        true_length = []
        mask1d = []
        minibatch_ids = []
        file_names = []
        max_len = 0
        for _ in range(batch_size):
            # get cached features
            id2f, id2r, fn = next(self.feature_grnerator)
            file_names.append(fn)
            # padding slot realizations
            corefs, _ = next(self.corefs_generator)
            hs = []
            fs = []
            rs = []
            es = []
            ids = []
            for coref in corefs.values():
                for realization in coref:
                    if not realization["isRepresentativeMention"]:
                        continue
                    id = realization["id"]
                    head = realization["headWord"]
                    hs.append(self.vocab.stoi[head])
                    entity_label = realization["entityType"]
                    es.append(self.entity_vocab.stoi[entity_label])
                    fs.append(id2f[id])  # [256]
                    rs.append(id2r[id])
                    ids.append(id)
            del id2f, id2r, corefs
            if len(hs) > max_len:
                max_len = len(hs)
            minibatch_hs.append(hs)
            minibatch_fs.append(fs)
            minibatch_rs.append(rs)
            minibatch_es.append(es)
            minibatch_ids.append(ids)

        # padding empty slot realization to max
        for i in range(batch_size):
            true_length.append(len(minibatch_hs[i]))
            need_len = max_len - len(minibatch_hs[i])
            mask1d.append([1. for _ in range(len(minibatch_hs[i]))] + [0. for _ in range(need_len)])
            if need_len > 0:
                minibatch_hs[i].extend([self.vocab.stoi[self.vocab.PADDING] for _ in range(need_len)])
                minibatch_fs[i].extend([torch.zeros(256) for _ in range(need_len)])
                minibatch_rs[i].extend([0. for _ in range(need_len)])
                minibatch_es[i].extend([self.entity_vocab.stoi[self.entity_vocab.PADDING] for _ in range(need_len)])
            minibatch_fs[i] = torch.stack(minibatch_fs[i], dim=0)

        minibatch_hs = torch.LongTensor(minibatch_hs).detach()
        minibatch_fs = torch.stack(minibatch_fs, dim=0).detach()
        minibatch_rs = torch.FloatTensor(minibatch_rs).detach()
        minibatch_es = torch.LongTensor(minibatch_es).detach()
        true_length = torch.LongTensor(true_length).detach()
        mask1d = torch.FloatTensor(mask1d).detach()
        # end_time = time.time()
        # print("spent", end_time - start_time, "s")
        return minibatch_hs, minibatch_fs, minibatch_rs, minibatch_es, true_length, mask1d, file_names, minibatch_ids


def debug_memory():
    import collections, gc, torch
    tensors = collections.Counter((str(o.device), o.dtype, tuple(o.shape))
                                  for o in gc.get_objects()
                                  if torch.is_tensor(o))
    for line in sorted(tensors.items()):
        print('{}\t{}'.format(*line))


def visualize_event_embedding():
    pass


def slot_coherence():
    pass


def generate_template_for_document():
    pass
