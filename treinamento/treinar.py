import os
import argparse
import logging
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def tokenize_function(examples, tokenizer, max_length=512):
    """
    Tokeniza os exemplos do dataset.
    
    Args:
        examples: Exemplos do dataset a serem tokenizados
        tokenizer: Tokenizador a ser utilizado
        max_length: Tamanho máximo da sequência
        
    Returns:
        Exemplos tokenizados
    """
    text_column = 'text' if 'text' in examples else list(examples.keys())[0]
    
    tokenized = tokenizer(
        examples[text_column],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_special_tokens_mask=True,
    )
    
    return tokenized

def main():
    parser = argparse.ArgumentParser(description="Script para fine-tuning do modelo distilGPT2")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="/Users/viniciussilvamoreira/Downloads/IA_suporte_tecnico/treinamento/dataset_convertido/",
        help="Caminho para o dataset processado"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./modelo",
        help="Diretório para salvar o modelo"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="distilgpt2",
        help="Nome do modelo base a ser utilizado"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Tamanho do batch para treinamento"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5,
        help="Taxa de aprendizado"
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=6,
        help="Número de épocas de treinamento"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="Tamanho máximo das sequências"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Semente para reprodutibilidade"
    )
    parser.add_argument(
        "--save_steps", 
        type=int, 
        default=500,
        help="Frequência para salvar checkpoints"
    )
    parser.add_argument(
        "--logging_steps", 
        type=int, 
        default=100,
        help="Frequência para logging"
    )
    parser.add_argument(
        "--warmup_steps", 
        type=int, 
        default=500,
        help="Número de passos de warmup para o scheduler"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="Valor para weight decay"
    )
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    logger.info(f"Carregando dataset de {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    
    logger.info(f"Carregando modelo {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    logger.info("Tokenizando dataset")
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=[col for col in dataset["train"].column_names if col != "text"],
        desc="Tokenizando dataset",
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_steps=args.save_steps,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        fp16=True,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )
    
    logger.info("Iniciando treinamento")
    train_result = trainer.train()
    
    logger.info(f"Salvando modelo em {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    logger.info("Avaliando modelo no conjunto de teste")
    eval_results = trainer.evaluate(tokenized_datasets["test"])
    trainer.log_metrics("test", eval_results)
    trainer.save_metrics("test", eval_results)
    
    logger.info("Treinamento concluído com sucesso!")
    
if __name__ == "__main__":
    main()