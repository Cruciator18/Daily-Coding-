import datasets
import datasets.utils.logging
from transformers import BertTokenizerFast


datasets.utils.logging.set_verbosity_error()

def load_and_prep_data(encoder_max_len=256, decoder_max_len=64):
    dataset = datasets.load_dataset("knkarthick/dialogsum")

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    def process_data_to_model_inputs(batch):
        inputs = tokenizer(batch["dialogue"], padding="max_length", truncation=True, max_length=encoder_max_len)
        outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_len)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()

       

        batch["labels"] = [
            [-100 if token == tokenizer.pad_token_id else token for token in labels] 
            for labels in batch["labels"]
        ]
        return batch

    train_data = dataset["train"].map(
        process_data_to_model_inputs, 
        batched=True, 
        remove_columns=["id", "dialogue", "summary", "topic"]
    )
    val_data = dataset["validation"].map(
        process_data_to_model_inputs, 
        batched=True, 
        remove_columns=["id", "dialogue", "summary", "topic"]
    )

    train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return train_data, val_data, tokenizer