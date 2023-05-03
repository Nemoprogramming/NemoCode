import random
random.seed(2023)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DataCollatorWithPadding, AutoTokenizer, TrainingArguments, Trainer
import data
import argparse
from modeling_t5_class import T5ForSequenceClassification
base = ''
def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
}

def get_data():
    return data.clean_data_defect()

def run(consolidated_ds, base=''):
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = T5ForSequenceClassification.from_pretrained(base, num_labels=2)
    def preprocess_function(record):
        return tokenizer(record["code"], truncation=True)
    tokenized_ds = consolidated_ds.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=50,
        weight_decay=0.01,
        evaluation_strategy="epoch"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.predict(tokenized_ds["test"])

def compare(consolidated_ds, model_str=''):
    tokenizer = AutoTokenizer.from_pretrained(model_str)
    def codet5_preprocess_function(record):
        return tokenizer(record["code"], truncation=True)
    tokenized_ds = consolidated_ds.map(codet5_preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = T5ForSequenceClassification.from_pretrained(model_str, num_labels=2)
    trainer = Trainer(model=model,
                        data_collator=data_collator,
                        compute_metrics=compute_metrics
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