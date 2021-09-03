from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor, SquadExample, SquadProcessor
from tqdm import tqdm

class SquadProcessorForMask(SquadProcessor):
    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:

                        answer = qa["answers"][0]
                        answer_text = answer["text"]
                        start_position_character = answer["answer_start"]
                        answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
        return examples


class SquadV1Processor(SquadProcessorForMask):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessorForMask):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"