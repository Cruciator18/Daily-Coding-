import evaluate
from transformers import EncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer
from BERT2BERT.data_setup import load_and_prep_data

def main():
    train_data, val_data, tokenizer = load_and_prep_data()

    model = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    model.generation_config.max_length = 64
    model.generation_config.min_length = 10
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.early_stopping = True
    model.generation_config.length_penalty = 2.0
    model.generation_config.num_beams = 4

    model.generation_config.decoder_start_token_id = tokenizer.bos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    rouge = evaluate.load("rouge")

    def compute_metrics(pred):
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.pad_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        rouge_output = rouge.compute(predictions=pred_str, references=label_str)
        return {
            "rouge2_fmeasure": round(rouge_output["rouge2"], 4),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir="./bert2bert-samsum",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        fp16=True,
        dataloader_num_workers=2,
        generation_num_beams=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()