import evaluate
import data
from transformers import AutoModelForSeq2SeqLM,Seq2SeqTrainingArguments,Seq2SeqTrainer,RobertaTokenizer,AutoTokenizer,AutoTokenizer,DataCollatorForSeq2Seq
import data
import evaluate
import argparse
import numpy as np
base = ''
curr_tokenizer = None
def get_data():
    return data.clean_data_review()

def compute_metrics(eval_pred):
    bleu = evaluate.load("bleu")
    predictions, labels = eval_pred
    decoded_preds = curr_tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, curr_tokenizer.pad_token_id)
    decoded_labels = curr_tokenizer.batch_decode(labels, skip_special_tokens=True)
    results = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    exact_match = evaluate.load("exact_match")
    results = exact_match.compute(references=decoded_labels, predictions=decoded_preds,
                                  ignore_case=True, ignore_punctuation=True)
    return {'bleu':results['bleu'],'EM':round(results["exact_match"], 2)}

def preprocess_function(examples):
    global base
    prefix = "summarize: "
    tokenizer = AutoTokenizer.from_pretrained(base)
    inputs = [prefix + doc for doc in examples["code"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["docstring"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def run(consolidated_ds, base='Salesforce/codet5-small'):
    global curr_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base)
    curr_tokenizer = tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(base)
    tokenized_ds = consolidated_ds.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model="Salesforce/codet5-small")
    training_args = Seq2SeqTrainingArguments(
                                            output_dir="./results",
                                            evaluation_strategy="epoch",
                                            learning_rate=2e-5,
                                            per_device_train_batch_size=8,
                                            per_device_eval_batch_size=8,
                                            weight_decay=0.01,
                                            num_train_epochs=5,
                                            predict_with_generate=True,
                                            fp16=True
                                            )
    trainer = Seq2SeqTrainer(
                            model=model,
                            args=training_args,
                            train_dataset=tokenized_ds["train"],
                            eval_dataset=tokenized_ds["test"],
                            tokenizer=tokenizer,
                            data_collator=data_collator,
                            compute_metrics=compute_metrics,
                        )

    trainer.train()
    trainer.predict(tokenized_ds["test"])

def compare(consolidated_ds, model_str=''):
    global curr_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    curr_tokenizer = tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_str)
    def preprocess_function(examples):
        prefix = "summarize: "
        inputs = [prefix + doc for doc in examples["code"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        labels = tokenizer(text_target=examples["docstring"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    tokenized_ds = consolidated_ds.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,model=model_str)
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        num_train_epochs=50,
        predict_with_generate=True,
        fp16=True
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.predict(tokenized_ds["test"])

def main():
    global base
    data = get_data()
    run(data, base)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-base", "--base", help = "Base model.")
    parser.add_argument('-baselines', '--baselines', nargs='+', default=[],help = "All baseline model checkpoints.")
    args = parser.parse_args()
    if args.base == False:
        print("Missing Base Model!")
    else:
        main()
        if args.baselines:
            for baseline in args.baselines:
                compare(data,baseline)