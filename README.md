# Open Domain Event Extraction Using Neural Latent Variable Models (ODEE)
This is the python3 code for the paper ["Open Domain Event Extraction Using Neural Latent Variable Models"](https://arxiv.org/abs/1906.06947) in ACL 2019.

## Prepare ELMo model
Modify the Line 24 and 25 in `cache_features.py`.

The fine-tune process need 2 * GTX 1080Ti, if the fine-tune process is costly or somehow failed 
to complete, please use the initial parameters in [allennlp](https://allennlp.org/elmo).

## Prepare Data and Train Model

The data is [HERE](https://drive.google.com/open?id=1KjL3mAxj9nmzqC75s2rNaT6x6CJBZZTj).

1. run the dataprocessor
2. run `sudo chown [YOUR_UERS] [PROCESSED_DIR]` and specify the directories in `setting.yaml` manually
2. run `pip install -r requirements.txt` to install required packages
3. run `python cache_features.py`
4. run `python train_avitm.py`
5. run `python generate_slot_topN.py`
6. run `python decode.py`
7. run `cd slotcoherence && ./run-oc.sh`
8. run `visualize_test.ipynb`

## Produced Data
1. `*.json.pt`: cached features of ODEE input
2. `*.json.answer`: decoded full results of a news group
3. `*.json.template`: decoded template of a news group
4. `*.json.events.topN`: decoded top-N events of a news group
5. `*.json.labeled`: labeled events of test split
6. `slotcoherence/slot_head_words.txt`: generated topN head words for each slot

## Cite
Please cite our ACL 2019 paper:
```bibtex
@proceedings{liu2019open,
    title={Open Domain Event Extraction Using Neural Latent Variable Models},
    author={Liu, Xiao and Huang, Heyan and Zhang, Yue},
    booktitle={Proceedings of Annual Meeting of the Association for Computational Linguistics},
    year={2019},
    publisher={Association for Computational Linguistics}
}
```