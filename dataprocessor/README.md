# ODEE Data Preprocessor
An tool to preprocess GN dataset into Staford CoreNLP XML output format 

python3 for data preprocessing

run `pip -r install reuqirements` to install requirements.

## Preprocess GN dataset

### Prepare Stanford CoreNLP
Download the [VERSION 3.9.1](https://stanfordnlp.github.io/CoreNLP/history.html),
and rewrite the variable `CORENLP_HOME` which indicates location of Stanford CoreNLP packages in `./setting.yaml`

### Run Preprocess Steps
```bash
Me@PC$ sudo /home/liuxiao/anaconda3/bin/python odee_preprocess.py /home/liuxiao/projects/schema/data_annotation/test_data parsed_test >& parsed_test.log &
Me@PC$ sudo /home/liuxiao/anaconda3/bin/python odee_preprocess.py /home/liuxiao/projects/schema/data_annotation/dev_data parsed_dev >& parsed_dev.log &
Me@PC$ sudo /home/liuxiao/anaconda3/bin/python odee_preprocess.py /home/liuxiao/projects/schema/data_annotation/unlabeled_data parsed_unlabeled >& parsed_unlabeled.log &
Me@PC$ sudo chown liuxiao parsed_test
Me@PC$ sudo chown liuxiao parsed_dev
Me@PC$ sudo chown liuxiao parsed_unlabeled
```

### Copy Labeled Data
```bash
Me@PC$ python copy_labeled.py /home/liuxiao/projects/schema/data_annotation/test_data parsed_test
Me@PC$ python copy_labeled.py /home/liuxiao/projects/schema/data_annotation/dev_data parsed_dev
```

## Prepare Full Text as Reference Corpus
```bash
Me@PC$ sudo /home/liuxiao/anaconda3/bin/python prepare_ref_corpus.py /home/liuxiao/projects/schema/data_annotation/test_data corpus.test &> corpus_test.log &
Me@PC$ sudo /home/liuxiao/anaconda3/bin/python prepare_ref_corpus.py /home/liuxiao/projects/schema/data_annotation/dev_data corpus.dev &> corpus_dev.log &
Me@PC$ sudo /home/liuxiao/anaconda3/bin/python prepare_ref_corpus.py /home/liuxiao/projects/schema/data_annotation/unlabeled_data corpus.unlabeled &> corpus_unlabeled.log &
Me@PC$ sudo chown liuxiao corpus.test
Me@PC$ sudo chown liuxiao corpus.dev
Me@PC$ sudo chown liuxiao corpus.unlabeled
Me@PC$ cat corpus.test corpus.dev corpus.unlabeled > corpus
```

## Produced Data
1. `*.txt`: original text
2. `*.json`: json-style ODEE input
3. `corpus.*`: tokenized full text corpus
