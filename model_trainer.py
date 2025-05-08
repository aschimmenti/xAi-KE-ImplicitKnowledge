from trl import SFTTrainer
from transformers import DataCollatorWithPadding, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from torch import torch
from sklearn.metrics import classification_report, balanced_accuracy_score
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

from tqdm import tqdm
import yaml
import random

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

explicit_path = config["explicit_data_path"]
implicit_path = config["implicit_data_path"]
seed = config["seed"]

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class ModelTrainer():
    def __init__(self, model_name, tokenizer_name, num_labels=5, lora_rank=256, lora_alpha=64, n_epochs=3):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.n_epochs = n_epochs
        
        # Load model and tokenizer
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, device_map="auto")
        self.tokenizer  = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Initialize LoRA configuration
        self.lora_config = LoraConfig(
            r = self.lora_rank, 
            target_modules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",],
            lora_alpha = self.lora_alpha,
            lora_dropout = 0.15, 
            bias = "none", 
            task_type = TaskType.SEQ_CLS,
        )
        self.model = get_peft_model(self.model, self.lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        # Set training arguments
        self.training_args = TrainingArguments(
            output_dir="./results",           
            eval_strategy = 'epoch',
            eval_steps = 50,
            learning_rate= 3e-5,
            lr_scheduler_type = "cosine_with_restarts", 
            logging_steps=10,
            weight_decay=0.01,
            num_train_epochs=self.n_epochs,
            fp16=True,
            report_to="none",
            run_name=f"{model_name}_r{lora_rank}_{n_epochs}e_alpha{lora_alpha}"
        )

        # Prepare data collator
        self.data_collator = DataCollatorWithPadding(self.tokenizer)
        self.metric = evaluate.load("recall")
        print(self.model.forward.__code__.co_varnames)

    
    def load_data(self, path):
        df = pd.read_csv(path)
        selected_occupations = {"actor", "film actor", "television actor", "stage actor", "film director"}
        filtered_df= df[(df["selected_property"] == "occupation") & (df["property_value"].isin(selected_occupations))]
        df = filtered_df[["original_text", "property_value"]]
        df = df.rename(columns={"original_text": "sentence", "property_value": "label"})
        guiding_prompt = "What does this person do for a living?"
        df['sentence'] = [f"Text: {text} \n Question: {guiding_prompt}" for text in df["sentence"]]
        df['label'] = df['label'].astype('category')
        df['target'] = df['label'].cat.codes
        LABEL_MAPPING = dict(enumerate(df['label'].cat.categories))
        
        X_train, X_temp, y_train, y_temp = train_test_split(df['sentence'], df['target'], test_size=0.6, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=42)
        
        train_data = Dataset.from_pandas(pd.DataFrame({"text": X_train, "label": y_train}))
        val_data = Dataset.from_pandas(pd.DataFrame({"text": X_val, "label": y_val}))
        test_data = Dataset.from_pandas(pd.DataFrame({"text": X_test, "label": y_test}))

        return train_data, val_data, test_data, LABEL_MAPPING

    def tokenize_data(self, train_data, val_data, test_data):
        def tokenize_function(examples):
            inputs = self.tokenizer(examples["text"], padding=True, truncation=True, return_tensors="pt")
            #print(inputs)
            inputs = inputs.to("cuda")
            return inputs

        tokenized_train = train_data.map(tokenize_function, batched=True).shuffle(seed=42)
        tokenized_val = val_data.map(tokenize_function, batched=True)
        tokenized_test = test_data.map(tokenize_function, batched=True)
        
        return tokenized_train, tokenized_val, tokenized_test


    def build_trainer(self, train_dataset, eval_dataset):        
        trainer = SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics
        )
        return trainer

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return self.metric.compute(predictions=predictions, references=labels, average="macro")

    def evaluate(self, sft_trainer, test_dataset):
        test_results = sft_trainer.predict(test_dataset)
        logits = test_results.predictions
        predictions = logits.argmax(axis=-1)
        
        # Softmax probabilities and confidence scores
        probabilities = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
        confidence_scores = probabilities.max(axis=-1)
        
        # Add predictions and confidence scores to dataset
        test_dataset_with_predictions = test_dataset.add_column("predictions", predictions)
        test_dataset_with_confidence = test_dataset_with_predictions.add_column("confidence", confidence_scores)
        test_df = test_dataset_with_confidence.to_pandas()
        
        return test_df

    
    def get_performance_metrics(self, df_test, LABEL_MAPPING):
        y_test = df_test["label"]
        y_pred = df_test["predictions"]

        cm = confusion_matrix(y_test, y_pred)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nMetrics:")
        
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        precision = precision_score(y_test, y_pred, average="macro")
        print(f"Precision: {precision:.4f}")
        
        metric_result = self.metric.compute(predictions=y_pred, references=y_test, average="macro")
        recall = metric_result['recall']
        print(f"Recall: {recall:.4f}")
        
        f1 = f1_score(y_test, y_pred, average="macro")
        print(f"F1 Score: {f1:.4f}")
        metrics = {
            "Balanced Accuracy": balanced_accuracy,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
            }

        print(LABEL_MAPPING)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=set(y_test), yticklabels=set(y_test))
        plt.title('Confusion Matrix Heatmap')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        return metrics
        