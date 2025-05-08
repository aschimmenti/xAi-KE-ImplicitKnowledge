from model_trainer import ModelTrainer
from metrics import MetricsTracker
from datasets import concatenate_datasets
import yaml


def run_experiment(model_name: str):
    tokenizer_name = model_name

    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    explicit_path = config["explicit_data_path"]
    implicit_path = config["implicit_data_path"]
    
    tracker = MetricsTracker()

    ## Train and test on explicit
    mode = "Train and test on explicit"
    trainer_class = ModelTrainer(model_name=model_name, tokenizer_name=tokenizer_name)
    train_data_exp, val_data_exp, test_data_exp, label_mapping = trainer_class.load_data(explicit_path)
    tokenized_train_exp, tokenized_val_exp, tokenized_test_exp = trainer_class.tokenize_data(train_data_exp, val_data_exp, test_data_exp)
    print(mode)
    print("After fine tuning")
    sft_trainer = trainer_class.build_trainer(tokenized_train_exp, tokenized_val_exp)
    sft_trainer.train()
    test_df = trainer_class.evaluate(sft_trainer, tokenized_test_exp)
    metrics = trainer_class.get_performance_metrics(test_df, label_mapping)
    tracker.store(mode, metrics)

    ## Train and test on implicit
    # del trainer_class
    del sft_trainer
    mode = "Train and test on implicit"
    trainer_class = ModelTrainer(model_name=model_name, tokenizer_name=tokenizer_name)
    train_data_imp, val_data_imp, test_data_imp, label_mapping = trainer_class.load_data(implicit_path)
    tokenized_train_imp, tokenized_val_imp, tokenized_test_imp = trainer_class.tokenize_data(train_data_imp, val_data_imp, test_data_imp)
    print(mode)
    print("After fine tuning")
    sft_trainer = trainer_class.build_trainer(tokenized_train_imp, tokenized_val_imp)
    sft_trainer.train()
    test_df = trainer_class.evaluate(sft_trainer, tokenized_test_imp)
    metrics = trainer_class.get_performance_metrics(test_df, label_mapping)
    tracker.store(mode, metrics)

    ## Train on explicit and implicit and test on both
    # del trainer_class
    del sft_trainer
    trainer_class = ModelTrainer(model_name=model_name, tokenizer_name=tokenizer_name)
    mode = "Testing trained on explicit and implicit on explicit"
    print(mode)
    print("After fine tuning")
    sft_trainer = trainer_class.build_trainer(concatenate_datasets([tokenized_train_imp, tokenized_train_exp]), concatenate_datasets([tokenized_train_imp, tokenized_train_exp]))
    sft_trainer.train()
    test_df_exp = trainer_class.evaluate(sft_trainer, tokenized_test_exp)
    metrics = trainer_class.get_performance_metrics(test_df_exp, label_mapping)
    tracker.store(mode, metrics)
    mode = "Testing trained on explicit and implicit on implicit"
    print(mode)
    test_df_imp = trainer_class.evaluate(sft_trainer, tokenized_test_imp)
    metrics = trainer_class.get_performance_metrics(test_df_imp, label_mapping)
    tracker.store(mode, metrics)

    ## Train on explicit and test on implicit
    # del trainer_class
    del sft_trainer
    trainer_class = ModelTrainer(model_name=model_name, tokenizer_name=tokenizer_name)
    mode = "Train on explicit and test on implicit"
    print(mode)
    sft_trainer = trainer_class.build_trainer(tokenized_train_exp, tokenized_val_exp)
    sft_trainer.train()
    test_df = trainer_class.evaluate(sft_trainer, tokenized_test_imp)
    metrics = trainer_class.get_performance_metrics(test_df, label_mapping)
    tracker.store(mode, metrics)

    ## Train on implicit and test on explicit
    # del trainer_class
    del sft_trainer
    trainer_class = ModelTrainer(model_name=model_name, tokenizer_name=tokenizer_name)
    mode = "Train on implicit and test on explicit"
    print(mode)
    sft_trainer = trainer_class.build_trainer(tokenized_train_imp, tokenized_val_imp)
    sft_trainer.train()
    test_df = trainer_class.evaluate(sft_trainer, tokenized_test_exp)
    metrics = trainer_class.get_performance_metrics(test_df, label_mapping)
    tracker.store(mode, metrics)

    # Display the results
    df = tracker.to_dataframe()
    print(df)
    tracker.save_to_csv(model_name)