import json
from tqdm import tqdm
import argparse
from pathlib import Path
import gzip


def main(config):
    output = {}
    qa_ids = []

    with gzip.GzipFile(config.file, 'r') as reader:
        # skip header
        content = reader.read().decode('utf-8').strip().split('\n')
        f = [json.loads(line) for line in content]

        header = f[0]
        f = f[1:]
        print("Processing the file:", header)
        dir = Path(config.file).parent
        split = header["header"]["split"] if "split" in header["header"] else header["header"]["mrqa_split"]
        output_file = dir / f'{header["header"]["dataset"]}-{split}-from-MRQA.json'
        print("Output will be stored in: ", output_file)
        if config.dry_run:
            return None
        n_qa = 0
        output["version"] = "1.1" # because there are no unanswerble questions
        output["data"] = [{"paragraphs": []}]
        for line in tqdm(f):
            paragraph = {}
            paragraph["context"] = line["context"]
            paragraph["qas"] = []
            for qa in line["qas"]:
                qa["id"] = qa["qid"]
                qa_ids.append(qa["id"])
                answers = []
                for a in qa["detected_answers"]:
                    answers.append({"text": a["text"], "answer_start": a["char_spans"][0][0]})
                qa["answers"] = answers
                paragraph["qas"].append(qa)
                del qa["qid"]
                del qa["question_tokens"]
                del qa["detected_answers"]
                n_qa += 1
            output["data"][0]["paragraphs"].append(paragraph)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=4)
    print("File is saved in:", output_file)
    print("The total number of QA pairs is:", n_qa, "(# of unique ids:", len(set(qa_ids)), ")")
    print()
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--file', type=str, default='')
    parser.add_argument('--dry_run', action='store_true', default=False)
    args = parser.parse_args()
    main(args)