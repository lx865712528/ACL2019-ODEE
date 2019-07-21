import torch

import yaml
from allennlp.modules.elmo import Elmo, batch_to_ids

from data import ParsedCorpus

if __name__ == "__main__":

    with open("setting.yaml", "r") as stream:
        setting = yaml.load(stream)

    base_dirs = [setting["parsed_data_path"]["test"],
                 setting["parsed_data_path"]["dev"],
                 setting["parsed_data_path"]["unlabeled"]]
    print("base_dirs are", base_dirs)

    corpus = ParsedCorpus(base_dirs)

    sentences_generator = corpus.get_single("sentences")
    corefs_generator = corpus.get_single("corefs")

    # if you are looking for example, please see https://allennlp.org/elmo
    options_file = "/path/to/options.json"
    weight_file = "path/to/weights.hdf5"
    encoder = Elmo(options_file, weight_file, 1, dropout=0)
    encoder.eval()
    encoder.cuda()

    for _ in range(len(corpus)):
        sentences, file_name = next(sentences_generator)
        corefs, _ = next(corefs_generator)
        save_name = file_name + ".pt"
        # preprocess all sentences in a document
        doc = []
        for sentence in sentences:
            sentence = [token["word"] for token in sentence["tokens"]]
            doc.append(sentence)
        character_ids = batch_to_ids(doc).cuda()
        # [sentence_num, sentence_len, 256]
        embeddings = encoder(character_ids)['elmo_representations'][0].detach().cpu().data

        # padding slot realizations
        max_rr = 0
        id2f = {}
        id2r = {}
        for coref in corefs.values():
            rr = len(coref)
            if rr > max_rr:
                max_rr = rr
            for realization in coref:
                id = realization["id"]
                sent_num = realization["sentNum"]
                head_index = realization["headIndex"]
                # [256]
                rep = embeddings[sent_num, head_index, :].detach()
                id2f[id] = rep
                id2r[id] = rr
        for id, rr in id2r.items():
            id2r[id] = rr / max_rr
        torch.save({"fs": id2f, "rs": id2r}, save_name)
        print("Processing %s to %s" % (file_name, save_name))

    print("Done!")
