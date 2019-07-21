import argparse
import os

import numpy as np
import torch
import yaml
from data import ParsedCorpus, HeadWordVocabulary, DataIterator
from model_avitm import Extractor
from tqdm import tqdm

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
parser.add_argument('--nogpu', action='store_true', default=False)  # do not use GPU acceleration
parser.add_argument('-p', '--model-path', type=str, default='models/default.pt')

args = parser.parse_args()

with open("setting.yaml", "r") as stream:
    setting = yaml.load(stream)

# default to use GPU, but have to check if GPU exists
if not args.nogpu:
    if torch.cuda.device_count() == 0:
        args.nogpu = True


def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)


def transform_counts(hs, vocab):
    hs = hs.numpy()
    counts = torch.FloatTensor([to_onehot(doc.astype('int'), len(vocab)) for doc in hs])
    counts[:, vocab.stoi[vocab.PADDING]] = 0.0
    counts[:, vocab.stoi[vocab.UNK]] = 0.0
    return counts


def make_data():
    base_dirs = [setting["parsed_data_path"]["test"],
                 setting["parsed_data_path"]["dev"],
                 setting["parsed_data_path"]["unlabeled"]]
    print("base_dirs are", base_dirs)
    corpus = ParsedCorpus(base_dirs)

    vocab = HeadWordVocabulary()
    if os.path.exists("./voc.txt"):
        vocab.load()
    else:
        vocab.make_vocabulary(corpus, "headWord")
        vocab.save()
    print("vocab length is", len(vocab.stoi))

    entity_vocab = HeadWordVocabulary()
    if os.path.exists("./evoc.txt"):
        entity_vocab.load("./evoc.txt")
    else:
        entity_vocab.make_vocabulary(corpus, "entityType")
        entity_vocab.save("./evoc.txt")
    print("entity label vocab length is", len(entity_vocab.stoi))

    data_iterator = DataIterator(corpus, vocab, entity_vocab)
    return data_iterator, vocab, entity_vocab


def make_model(vocab, entity_vocab):
    global model
    net_arch = args
    net_arch.num_input = len(vocab)
    net_arch.entity_types = len(entity_vocab)
    model = Extractor(net_arch)
    if not args.nogpu:
        model = model.cuda()


def make_optimizer():
    global optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(args.momentum, 0.999))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    else:
        assert False, 'Unknown optimizer {}'.format(args.optimizer)


def train(iterator, vocab):
    for epoch in range(args.num_epoch):
        iterator.reset()
        loss_epoch = 0.0
        model.train()  # switch to training mode
        max_step = (len(iterator) + args.batch_size - 1) // args.batch_size
        with tqdm(total=max_step, desc='Epoch %d Progress' % (epoch + 1)) as pbar:
            for iter_no in range(max_step):
                hs, fs, rs, _, _, mask, _, _ = iterator.get_minibatch(args.batch_size)
                hcounts = transform_counts(hs, vocab)
                feas = torch.cat([fs, rs.unsqueeze(-1)], dim=-1)
                if not args.nogpu:
                    hcounts = hcounts.cuda()
                    feas = feas.cuda()
                    mask = mask.cuda()
                _, _, _, loss = model(hcounts, feas, mask, compute_loss=True)
                # optimize
                optimizer.zero_grad()  # clear previous gradients
                loss.backward()  # backprop
                optimizer.step()  # update parameters
                # report
                loss_epoch += loss.item()  # add loss to loss_epoch
                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
                pbar.update(1)
        print('Epoch {}, loss={}'.format(epoch + 1, loss_epoch / max_step))
        model.save_cpu_model(args.model_path)
    model.save_cpu_model(args.model_path)
    print("Done!")


def print_top_words(beta, feature_names, n_top_words=10):
    print('---------------Printing the Topics------------------')
    for i in range(len(beta)):
        line = " ".join([feature_names[j] for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        print('     {}'.format(line))
    print('---------------End of Topics------------------')


if __name__ == '__main__':
    data_iterator, vocab, entity_vocab = make_data()
    make_model(vocab, entity_vocab)
    make_optimizer()
    train(data_iterator, vocab)
    emb = model.get_unnormalized_phi()  # [K, V]
    print_top_words(emb, vocab.itos)
