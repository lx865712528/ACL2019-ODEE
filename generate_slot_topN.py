import argparse

import yaml
from data import ParsedCorpus, HeadWordVocabulary
from model_avitm import Extractor

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--l1-units', type=int, default=100)
parser.add_argument('-s', '--l2-units', type=int, default=100)
parser.add_argument('-d', '--feature-dim', type=int, default=256)
parser.add_argument('-t', '--num-topic', type=int, default=30)
parser.add_argument('-b', '--batch-size', type=int, default=200)
parser.add_argument('-o', '--optimizer', type=str, default='Adam')
parser.add_argument('-r', '--learning-rate', type=float, default=0.002)
parser.add_argument('-m', '--momentum', type=float, default=0.99)
parser.add_argument('-e', '--num-epoch', type=int, default=80)
parser.add_argument('-q', '--init-mult', type=float, default=1.0)  # multiplier in initialization of decoder weight
parser.add_argument('-v', '--variance', type=float, default=0.995)  # default variance in prior normal
parser.add_argument('-p', '--model-path', type=str, default='models/default.pt')

args = parser.parse_args()


def save_top_words(beta, feature_names, n_top_words=100, save_path="slotcoherence/slot_head_words.txt"):
    with open(save_path, "w", encoding="utf-8") as f:
        for i in range(len(beta)):
            line = " ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]])
            f.write('%s\n' % line)


with open("setting.yaml", "r") as stream:
    setting = yaml.load(stream)

base_dirs = [setting["parsed_data_path"]["test"],
             setting["parsed_data_path"]["dev"],
             setting["parsed_data_path"]["unlabeled"]]
print("base_dirs are", base_dirs)
corpus = ParsedCorpus(base_dirs)
vocab = HeadWordVocabulary()
vocab.load()
entity_vocab = HeadWordVocabulary()
entity_vocab.load("./evoc.txt")
net_arch = args
net_arch.num_input = len(vocab)
net_arch.nogpu = True
model = Extractor(net_arch)
model.load_cpu_model(args.model_path)
model.eval()
emb = model.get_unnormalized_phi()  # [K, V]
save_top_words(emb, vocab.itos)
print("Done!")
