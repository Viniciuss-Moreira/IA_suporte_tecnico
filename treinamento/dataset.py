import json
import os
from typing import Optional, Dict, List, Union, Any
from datasets import Dataset, DatasetDict
import random

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def jsonl_to_dataset(
    jsonl_path: str, 
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> DatasetDict:

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("as proporções train_ratio, val_ratio e test_ratio devem somar 1")
    
    data = load_jsonl(jsonl_path)
    
    random.seed(seed)
    random.shuffle(data)
    
    total = len(data)
    train_idx = int(train_ratio * total)
    val_idx = train_idx + int(val_ratio * total)
    
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset_dict

def save_dataset(dataset_dict: DatasetDict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    dataset_dict.save_to_disk(output_dir)
    print(f"Dataset salvo em {output_dir}")

if __name__ == "__main__":
    jsonl_path = "/Users/viniciussilvamoreira/Downloads/IA_suporte_tecnico/dados/tratados/dataset_artificial.jsonl"
    output_dir = "/Users/viniciussilvamoreira/Downloads/IA_suporte_tecnico/treinamento/dataset_convertido/"
    
    dataset = jsonl_to_dataset(
        jsonl_path=jsonl_path,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    print(f"dataset convertido")
    print(f"conjunto de treinamento: {len(dataset['train'])} exemplos")
    print(f"conjunto de validação: {len(dataset['validation'])} exemplos")
    print(f"conjunto de teste: {len(dataset['test'])} exemplos")
    
    print(f"colunas do dataset: {dataset['train'].column_names}")
    print(f"exemplo do dataset: {dataset['train'][0]}")
    
    save_dataset(dataset, output_dir)