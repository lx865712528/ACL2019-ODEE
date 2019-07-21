import argparse
import json
import pickle

import torch
import yaml
from data import ParsedCorpus, HeadWordVocabulary, DataIterator
from model_avitm import Extractor
from torch.distributions import MultivariateNormal
from torch.nn import functional as F
from tqdm import tqdm
from train_avitm import transform_counts

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

base_dirs = [setting["parsed_data_path"]["test"],
             setting["parsed_data_path"]["dev"],
             setting["parsed_data_path"]["unlabeled"]]
print("base_dirs are", base_dirs)

threshold = 0.5

corpus = ParsedCorpus(base_dirs)
vocab = HeadWordVocabulary()
vocab.load()
entity_vocab = HeadWordVocabulary()
entity_vocab.load("./evoc.txt")
net_arch = args
net_arch.num_input = len(vocab)
model = Extractor(net_arch)
model.load_cpu_model(args.model_path)
model.cuda()
model.eval()

iterator = DataIterator(corpus, vocab, entity_vocab)
iterator.reset()

slot_word_dist = F.log_softmax(torch.FloatTensor(model.get_unnormalized_phi()), dim=-1)  # tensor [K, V]
assert torch.isnan(slot_word_dist).sum().item() == 0
slot_mean_dist = torch.FloatTensor(model.get_beta_mean())  # tensor [K, D + 1]
slot_stdvar_dist = torch.FloatTensor(model.get_beta_logvar()).exp().sqrt()  # tensor [K, D + 1]
if not args.nogpu:
    slot_word_dist = slot_word_dist.cuda()
    slot_mean_dist = slot_mean_dist.cuda()
    slot_stdvar_dist = slot_stdvar_dist.cuda()
dists = [MultivariateNormal(loc=slot_mean_dist[k],
                            covariance_matrix=torch.diag_embed(slot_stdvar_dist[k]))
         for k in range(args.num_topic)]

max_step = (len(iterator) + args.batch_size - 1) // args.batch_size
with tqdm(total=max_step, desc='Forwarding minibatches') as pbar:
    for iter_no in range(max_step):
        hs, fs, rs, _, lens, mask, fns, ids = iterator.get_minibatch(args.batch_size)
        hcounts = transform_counts(hs, vocab)
        feas = torch.cat([fs, rs.unsqueeze(-1)], dim=-1)
        if not args.nogpu:
            hcounts = hcounts.cuda()
            feas = feas.cuda()
            mask = mask.cuda()
        event_types, posterior_means, posterior_vars = model(hcounts, feas, mask, compute_loss=False)  # [batch_size, D]
        ps = (F.softmax(model.s_fc(event_types), dim=-1) + 1e-10).log()  # [batch_size, K] mixture probability
        padded_logps = torch.stack([ps[:, k].unsqueeze(-1) +
                                    dists[k].log_prob(feas) +
                                    slot_word_dist[k][hs]
                                    for k in range(args.num_topic)], dim=-1).cpu()  # [batch_size, SEQ_LEN, K]
        assert torch.isnan(padded_logps).sum().item() == 0
        # iterating over batch_size / documents
        for i in range(hs.shape[0]):
            event = event_types[i]  # tensor [D]
            posterior_mean = posterior_means[i]  # tensor [D]
            true_len = lens[i]  # int
            h = hs[i, :true_len]  # tensor [true_len]
            fn = fns[i]  # str
            realization_id = ids[i]  # list [true_len]
            logp = padded_logps[i, :true_len]  # tensor [true_len, K]
            pp = torch.softmax(logp, dim=-1)

            with open(fn, "r", encoding="utf-8") as f:
                data_dict = json.load(f)
            id2entity = {
                entity_json["id"]: entity_json for chain in data_dict["corefs"].values() for entity_json in chain
            }
            id2weight = {
                entity_json["id"]: len(chain) for chain in data_dict["corefs"].values() for entity_json in chain
            }

            # dump pickle
            save_file_name = fn + ".answer"
            answer_dict = {
                "event": event.cpu().data.numpy(),
                "mean": posterior_mean.cpu().data.numpy(),
                "slot_realizations": [{
                    "id": id,
                    "probs": probs.data.numpy(),
                    "slot": torch.argmax(probs).item(),  # mle decoding
                    "headWord": vocab.itos[word_index],
                    "text": id2entity[id]["text"],
                    "entityType": id2entity[id]["entityType"],
                    "predicates": id2entity[id]["predicates"],
                } for id, word_index, probs in zip(realization_id, h, pp) if
                    probs[torch.argmax(probs).item()] > threshold + 1e-7]  # cut by threshold
            }
            id2answer = {
                x["id"]: x for x in answer_dict["slot_realizations"]
            }
            with open(fn + ".answer", "wb") as f:
                pickle.dump(answer_dict, f)

            # dump extracted template
            template = {}
            for slot in answer_dict["slot_realizations"]:
                entity_id = str(slot["id"])
                slot_id = str(slot["slot"])
                head_word = slot["headWord"]
                entity_label = slot["entityType"]
                text = slot["text"]
                if slot_id not in template:
                    template[slot_id] = []
                template[slot_id].append("%s (%s) #%s @%s " % (text, head_word, entity_label, entity_id))
            with open(fn + ".template", "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2, sort_keys=True, ensure_ascii=False)

            # dump extracted events
            trigger_dict = {}
            trigger_weight = {}
            for chain in data_dict["corefs"].values():
                all_predicates = set()
                eid = -1
                for realization in chain:
                    predicates = realization["predicates"]
                    for predicate in predicates:
                        all_predicates.add(predicate["lemma"].lower())
                    if realization["isRepresentativeMention"]:
                        eid = realization["id"]
                if eid == -1 or not (eid in id2answer):
                    continue
                for predicate in all_predicates:
                    if predicate not in trigger_dict:
                        trigger_dict[predicate] = []
                        trigger_weight[predicate] = 0
                    trigger_dict[predicate].append(id2answer[eid])
                    trigger_weight[predicate] += 1
            # get top-3 events
            predicate_lists = list(sorted(trigger_weight.keys(), key=lambda x: -trigger_weight[x]))[:3]
            events = {}
            for idx, predicate in enumerate(predicate_lists):
                new_event = {}
                entities = trigger_dict[predicate]
                new_event["trigger"] = predicate
                new_event["slots"] = {}
                for entity in entities:
                    entity_id = str(entity["id"])
                    slot_id = str(entity["slot"])
                    head_word = entity["headWord"]
                    entity_label = entity["entityType"]
                    text = entity["text"]
                    if slot_id not in new_event["slots"]:
                        new_event["slots"][slot_id] = []
                    new_event["slots"][slot_id].append("%s (%s) #%s @%s " % (text, head_word, entity_label, entity_id))
                events["No.%d atomic event" % (idx + 1)] = new_event
            with open(fn + ".events.topN", "w", encoding="utf-8") as f:
                json.dump(events, f, indent=2, sort_keys=True, ensure_ascii=False)

        pbar.update(1)
