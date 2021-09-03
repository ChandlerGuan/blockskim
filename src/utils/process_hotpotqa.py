import os

import tqdm
from datasets import load_dataset, load_from_disk

import json


def create_example_dict(context, answer_start, answer, id, question,title,fact_start_list,fact_list):
    return {
        "title":title,
        "paragraphs":
            [
                {
                    "context": context,
                    "qas":
                        [
                            {
                                "id": id,
                                "question": question,
                                "answers":
                                    [
                                        {
                                        "answer_start": answer_start,
                                        "text":answer
                                        },
                                    ],
                                "supporting_facts": 
                                    [
                                        {
                                            "fact_start": fact_start,
                                            "text": fact_text
                                        }
                                        for fact_start, fact_text in zip(fact_start_list, fact_list)
                                    ]
                            },
                        ]
                },
            ],
    }

def add_yes_no(string):
    # Allow model to explicitly select yes/no from text (location front, avoid truncation)
    return " ".join(["yes", "no", string])


def convert_hotpot_to_squad_format(
    example, gold_paras_only=True
):

    raw_contexts = example["context"]
    supporting_facts_title = example["supporting_facts"]['title']
    supporting_facts_sent_id = example["supporting_facts"]['sent_id']
    raw_contexts_title = example["context"]['title']
    raw_contexts_sentences = example["context"]['sentences']

    fact_start_list = []
    fact_list = []
    for i,fact_title in enumerate(supporting_facts_title):
        sentence_index=supporting_facts_sent_id[i]
        paragraph_index=raw_contexts_title.index(fact_title)
        try:
            fact_list.append(raw_contexts_sentences[paragraph_index][sentence_index])
        except:
            print(example)
            print("hh, what a bug")

    if gold_paras_only:
        context=[]
        for title,paragraph in zip(raw_contexts_title,raw_contexts_sentences):
            if title in supporting_facts_title:
                context.append(paragraph)
    else:
        context=example["context"]['sentences']

    for i in range(len(context)):
        context[i]="".join(context[i]) # 每个句子的开头本来就带着空格
    context=" ".join(context)
    context = add_yes_no(context)

    answer = example["answer"]
    answer_start = context.index(answer) if answer in context else -1

    for fact in fact_list:
        fact_start_list.append(context.index(fact))

    new_dict= create_example_dict(
                context=context,
                answer_start=answer_start,
                answer=answer,
                id=example['id'],  # SquadExample.__repr__ only accepts type==str
                question=example["question"],
                title="_".join(supporting_facts_title,),
                fact_start_list=fact_start_list,
                fact_list=fact_list
            )



    return new_dict

if __name__ == "__main__":
    hotpot_qa_distractor = load_dataset("hotpot_qa","distractor")

    converted_datasets = hotpot_qa_distractor.map(convert_hotpot_to_squad_format,False,num_proc=1,load_from_cache_file=True)

    converted_datasets = converted_datasets.remove_columns(['id','question','answer','type','level','supporting_facts','context'])

    print(converted_datasets)

    save_path = "./validation.json"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_dataset=converted_datasets['train']
    # using datasets to_json API causes un-loadable json file, parse the json file as dict manually instead
    # train_dataset_json = train_dataset.to_json('datasets/hotpotqa/gold_train.json', lines=False, orient='table')    
    train_dataset_dict = train_dataset.to_dict()
    train_dataset_dict = [{'title': title,'paragraphs': paragraphs} for title, paragraphs in zip(train_dataset_dict['title'],train_dataset_dict['paragraphs'])]
    train_dataset_dict = {'data':train_dataset_dict}
    with open('datasets/hotpotqa/gold_train.json','w') as output_file:
        json.dump(train_dataset_dict, output_file)

    validation_dataset=converted_datasets['validation']
    validation_dataset_json = validation_dataset.to_json('datasets/hotpotqa/gold_validation.json',lines=False,orient='table')