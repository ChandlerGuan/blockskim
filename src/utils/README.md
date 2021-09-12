# `process_hotpotqa.py`
This is a preprocess script for HotpotQA dataset with evidence. It transforms HotpotQA dataset from datasets library into SQuAD format. This enables reuse of `run_squad.py` code without extra effort. Supporting facts are encoded in similar format as answers in the dataset.
- `gold_paras_only` arg. to select only answer and supporting fact paragraphs into context.