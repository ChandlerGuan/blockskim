# `process_hotpotqa.py`
This preprocess script is contributed by Zhengyi Li. It transforms HotpotQA dataset from datasets library into SQuAD format. This enables reuse of `run_squad.py` code without extra effort. Supporting facts are encoded in similar format as answers in the dataset.
- `gold_paras_only` arg. to select only answer and supporting fact paragraphs into context.