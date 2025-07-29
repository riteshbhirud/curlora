#!/usr/bin/env python3
"""
Complete testing script for CURLoRA vs LoRA comparison - FIXED VERSION
Handles Conv1D layers properly for GPT-2
"""

import torch
import torch.nn as nn
import numpy as np
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
# Use AdamW from torch.optim for newer transformers versions
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW
from tqdm import tqdm
import gc
import argparse
import json
import time
from pathlib import Path

# Import custom modules - only the base classes we need
try:
    from curlora import CURModule
    from lora import LoRALayer
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the directory containing curlora.py, lora.py")
    print("Current working directory files:")
    import os
    print([f for f in os.listdir('.') if f.endswith('.py')])
    raise

# Helper classes to handle Conv1D layers in GPT-2
class Conv1DWithLoRA(torch.nn.Module):
    def __init__(self, conv1d, rank, alpha):
        super().__init__()
        self.conv1d = conv1d
        self.rank = rank
        self.alpha = alpha
        
        # Conv1D weight shape is (in_features, out_features)
        in_features = conv1d.weight.shape[0]
        out_features = conv1d.nf
        
        # Create LoRA matrices
        self.lora_A = torch.nn.Parameter(torch.empty(in_features, rank))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_features))
        
    def forward(self, x):
        # Original Conv1D forward
        original_out = self.conv1d(x)
        # LoRA addition
        lora_out = self.alpha * (x @ self.lora_A @ self.lora_B)
        return original_out + lora_out

class Conv1DWithCURLoRA(torch.nn.Module):
    def __init__(self, conv1d, rank, alpha):
        super().__init__()
        self.conv1d = conv1d
        self.alpha = alpha
        
        # Get weight matrix from Conv1D (shape: in_features, out_features)
        weight = conv1d.weight.data
        
        # Apply CUR decomposition
        C, U, R = self.cur_decomposition(weight, rank)
        
        self.C = C
        self.R = R  
        self.U = torch.nn.Parameter(U)
        
    def cur_decomposition(self, A, c):
        """Simplified CUR decomposition for Conv1D weights"""
        r = c
        # Compute selection probabilities
        column_norms_squared = torch.sum(A**2, axis=0)
        row_norms_squared = torch.sum(A**2, axis=1)
        total_sum_squares = torch.sum(column_norms_squared)
        column_probs = column_norms_squared / total_sum_squares
        row_probs = row_norms_squared / total_sum_squares
        
        # Select indices (simplified version)
        inverted_col_P = (1 / (column_probs + 0.001)).float()
        col_probs_norm = inverted_col_P / inverted_col_P.sum()
        if col_probs_norm.device.type == "cuda":
            col_probs_norm = col_probs_norm.cpu().numpy()
        else:
            col_probs_norm = col_probs_norm.numpy()
        selected_columns = np.random.choice(len(col_probs_norm), size=c, replace=True, p=col_probs_norm)
        
        inverted_row_P = (1 / (row_probs + 0.001)).float()
        row_probs_norm = inverted_row_P / inverted_row_P.sum()
        if row_probs_norm.device.type == "cuda":
            row_probs_norm = row_probs_norm.cpu().numpy()
        else:
            row_probs_norm = row_probs_norm.numpy()
        selected_rows = np.random.choice(len(row_probs_norm), size=r, replace=True, p=row_probs_norm)
        
        C = A[:, selected_columns]
        R = A[selected_rows, :]
        U = torch.zeros(C.shape[1], R.shape[0]).to(A.device)
        
        return C, U, R
        
    def forward(self, x):
        # Original Conv1D forward
        original_out = self.conv1d(x)
        # CURLoRA addition
        W_approx = torch.matmul(torch.matmul(self.C, self.U), self.R)
        curlora_out = self.alpha * (x @ W_approx)
        return original_out + curlora_out

# Define evaluation functions locally to avoid utils.py import issues
def calculate_perplexity(model, tokenizer, text, device='cpu'):
    """Calculate perplexity on given text"""
    if not text.strip():
        print("Warning: Empty text encountered")
        return float('inf')
    
    encodings = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=1000)
    input_ids = encodings.input_ids.to(device)
    
    if input_ids.numel() == 0:
        print(f"Warning: Empty input_ids for text: {text[:100]}...")
        return float('inf')
    
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        
    loss = outputs.loss
    perplexity = torch.exp(loss)
    
    try:
        return perplexity.item()
    finally:
        del input_ids, target_ids, outputs
        if device == "cuda":
            torch.cuda.empty_cache()
        _ = gc.collect()

def evaluate_sst2(model, tokenizer, dataset, device):
    """Evaluate on SST-2 dataset"""
    dataset = dataset["validation"]
    model.eval()
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc="Evaluating SST-2"):
        inputs = tokenizer.encode(example["sentence"], return_tensors="pt",
                                  truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        
        predicted = torch.argmax(outputs.logits[:, -1, :]).item()
        correct += (predicted == example["label"])
        total += 1
    
    accuracy = correct / total
    return accuracy

def evaluate_mrpc(model, tokenizer, dataset, device):
    """Evaluate on MRPC dataset"""
    dataset = dataset["validation"]
    model.eval()
    
    correct = 0
    total = 0
    
    for example in tqdm(dataset, desc="Evaluating MRPC"):
        inputs = tokenizer.encode(example["sentence1"], example["sentence2"], return_tensors="pt",
                                  truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        
        predicted = torch.argmax(outputs.logits[:, -1, :]).item()
        correct += (predicted == example["label"])
        total += 1
    
    accuracy = correct / total
    return accuracy

def evaluate_sentiment(model, tokenizer, dataset, device):
    """Evaluate on sentiment dataset - using SST-2 as a proxy for sentiment analysis"""
    # Since sentiment140 is not available, we'll use SST-2 as a sentiment task
    # This maintains the same evaluation structure but with a working dataset
    return evaluate_sst2(model, tokenizer, dataset, device)

class ModelTester:
    def __init__(self, model_name="gpt2-large", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.max_len = 512
        self.lr = 2.5e-4
        self.results = {}
        
        # Load datasets (avoiding sentiment140 which has issues)
        print("Loading datasets...")
        self.mrpc_dataset = load_dataset("glue", "mrpc")
        self.sst_dataset = load_dataset("glue", "sst2") 
        # Use a different sentiment dataset that works, or use SST-2 as sentiment proxy
        print("Note: Using SST-2 as sentiment analysis task (sentiment140 is deprecated)")
        self.sentiment_dataset = self.sst_dataset  # Use SST-2 as sentiment task
        
        # Load wikitext for perplexity
        wikidataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        
        # Prepare wikitext for perplexity calculation
        txt = wikidataset["text"]
        txt = [s for s in txt if s != '']
        self.wiki_text = "".join(txt[:50])  # Use smaller subset for CPU
        
    def load_base_model(self):
        """Load the base model and tokenizer"""
        print(f"Loading base model: {self.model_name}")
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Handle device placement
        if self.device == "cuda" and torch.cuda.is_available():
            model = model.to(self.device)
        else:
            model = model.to("cpu")
            self.device = "cpu"  # Update device if CUDA not available
            
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def apply_adaptation(self, model, adaptation_type="lora", rank=8, alpha=1):
        """Apply LoRA or CURLoRA to the model"""
        # Freeze base model parameters
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"Applying {adaptation_type} to GPT-2 Conv1D layers...")
        adaptation_count = 0
        
        # For GPT-2, look for all transformer blocks
        for i, block in enumerate(model.transformer.h):
            print(f"Processing transformer block {i}")
            print(f"  c_attn type: {type(block.attn.c_attn)}")
            
            if adaptation_type == "lora":
                block.attn.c_attn = Conv1DWithLoRA(block.attn.c_attn, rank, alpha)
            elif adaptation_type == "curlora":
                block.attn.c_attn = Conv1DWithCURLoRA(block.attn.c_attn, rank, alpha)
            adaptation_count += 1
        
        print(f"Applied {adaptation_type} to {adaptation_count} attention layers")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters after {adaptation_type}: {total_params:,}")
        
        return model, total_params
    
    def train_on_task(self, model, tokenizer, dataset_name, num_classes, num_epochs=3, batch_size=8):
        """Train the model on a specific task"""
        print(f"\nTraining on {dataset_name}...")
        
        # Prepare task-specific classification head
        torch.manual_seed(1311)
        in_features = 1280  # GPT-2 Large
            
        original_lm_head = model.lm_head
        model.lm_head = torch.nn.Linear(in_features=in_features, out_features=num_classes).to(self.device)
        
        # Get dataset
        if dataset_name == "mrpc":
            train_dataset = self.mrpc_dataset["train"]
            train_limit = 100  # Smaller limit for CPU testing
        elif dataset_name == "sst2":
            train_dataset = self.sst_dataset["train"]
            train_limit = 100  # Smaller limit for CPU testing
        elif dataset_name == "sentiment":
            # Use SST-2 as sentiment task since sentiment140 is deprecated
            train_dataset = self.sst_dataset["train"]
            train_limit = 100  # Smaller limit for CPU testing
            
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=self.lr)
        num_training_steps = num_epochs * (train_limit // batch_size)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=num_training_steps)
        
        # Training loop
        model.train()
        start_time = time.time()
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for i in tqdm(range(0, train_limit, batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch = train_dataset[i:i+batch_size]
                
                # Prepare inputs based on task
                if dataset_name == "mrpc":
                    inputs = tokenizer(batch["sentence1"], batch["sentence2"], 
                                     return_tensors="pt", truncation=True, 
                                     padding=True, max_length=self.max_len).to(self.device)
                    labels = torch.LongTensor(batch["label"]).to(self.device)
                elif dataset_name in ["sst2", "sentiment"]:
                    inputs = tokenizer(batch["sentence"], return_tensors="pt",
                                     truncation=True, padding=True, 
                                     max_length=self.max_len).to(self.device)
                    labels = torch.LongTensor(batch["label"]).to(self.device)
                
                optimizer.zero_grad()
                outputs = model(**inputs)["logits"][:, -1, :]
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Memory cleanup
                del inputs, outputs, labels, loss
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                _ = gc.collect()
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}, Average loss: {avg_loss:.6f}")
        
        training_time = time.time() - start_time
        
        # Save the task head and restore original
        task_head = model.lm_head
        model.lm_head = original_lm_head
        
        return task_head, training_time
    
    def evaluate_model(self, model, tokenizer, task_heads):
        """Evaluate the model on all tasks"""
        print("\nEvaluating model...")
        results = {}
        
        # Test each task
        for task_name, head in task_heads.items():
            print(f"\nEvaluating on {task_name}...")
            original_head = model.lm_head
            model.lm_head = head
            
            if task_name == "mrpc":
                # Use smaller subset for CPU evaluation
                small_dataset = {"validation": self.mrpc_dataset["validation"].select(range(50))}
                accuracy = evaluate_mrpc(model, tokenizer, small_dataset, self.device)
            elif task_name == "sst2":
                small_dataset = {"validation": self.sst_dataset["validation"].select(range(50))}
                accuracy = evaluate_sst2(model, tokenizer, small_dataset, self.device)
            elif task_name == "sentiment":
                small_dataset = {"validation": self.sst_dataset["validation"].select(range(50))}
                accuracy = evaluate_sentiment(model, tokenizer, small_dataset, self.device)
            
            results[task_name] = accuracy
            print(f"{task_name.upper()} Accuracy: {accuracy:.4f}")
            
            model.lm_head = original_head
        
        # Test perplexity with original language modeling head
        print("\nCalculating perplexity...")
        perplexity = calculate_perplexity(model, tokenizer, self.wiki_text, self.device)
        results["perplexity"] = perplexity
        print(f"Perplexity: {perplexity:.2f}")
        
        return results
    
    def test_continual_learning(self, adaptation_type="lora", rank=8, alpha=1):
        """Test continual learning scenario"""
        print(f"\n{'='*50}")
        print(f"Testing {adaptation_type.upper()} - Continual Learning")
        print(f"Tasks: MRPC -> SST-2 -> Sentiment (SST-2 proxy)")
        print(f"{'='*50}")
        
        model, tokenizer = self.load_base_model()
        model, total_params = self.apply_adaptation(model, adaptation_type, rank, alpha)
        
        # Calculate initial perplexity
        initial_perplexity = calculate_perplexity(model, tokenizer, self.wiki_text, self.device)
        print(f"Initial Perplexity: {initial_perplexity:.2f}")
        
        task_heads = {}
        training_times = {}
        
        # Sequential training on tasks
        tasks = [
            ("mrpc", 2),
            ("sst2", 2), 
            ("sentiment", 2)  # Using SST-2 as sentiment task, so also 2 classes
        ]
        
        for task_name, num_classes in tasks:
            head, train_time = self.train_on_task(model, tokenizer, task_name, num_classes)
            task_heads[task_name] = head
            training_times[task_name] = train_time
        
        # Evaluate on all tasks
        accuracies = self.evaluate_model(model, tokenizer, task_heads)
        
        # Compile results
        results = {
            "adaptation_type": adaptation_type,
            "trainable_parameters": total_params,
            "initial_perplexity": initial_perplexity,
            "final_perplexity": accuracies["perplexity"],
            "training_times": training_times,
            "accuracies": {k: v for k, v in accuracies.items() if k != "perplexity"}
        }
        
        return results
    
    def run_comparison(self, rank=8, alpha=1, save_results=True):
        """Run complete comparison between LoRA and CURLoRA"""
        print(f"Running comparison on {self.model_name}")
        print(f"Rank: {rank}, Alpha: {alpha}")
        
        # Test LoRA
        lora_results = self.test_continual_learning("lora", rank, alpha)
        
        # Clean up memory
        if self.device == "cuda":
            torch.cuda.empty_cache()
        _ = gc.collect()
        
        # Test CURLoRA  
        curlora_results = self.test_continual_learning("curlora", rank, alpha)
        
        # Combine results
        comparison_results = {
            "model_name": self.model_name,
            "rank": rank,
            "alpha": alpha,
            "lora": lora_results,
            "curlora": curlora_results
        }
        
        # Print summary
        self.print_comparison_summary(comparison_results)
        
        # Save results
        if save_results:
            self.save_results(comparison_results)
        
        return comparison_results
    
    def print_comparison_summary(self, results):
        """Print a summary comparison"""
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        lora = results["lora"]
        curlora = results["curlora"]
        
        print(f"Model: {results['model_name']}")
        print(f"Rank: {results['rank']}, Alpha: {results['alpha']}")
        print()
        
        # Parameter efficiency
        print("PARAMETER EFFICIENCY:")
        print(f"LoRA trainable parameters:    {lora['trainable_parameters']:,}")
        print(f"CURLoRA trainable parameters: {curlora['trainable_parameters']:,}")
        compression_ratio = lora['trainable_parameters'] / curlora['trainable_parameters']
        print(f"Compression ratio: {compression_ratio:.2f}x")
        print()
        
        # Performance comparison
        print("TASK PERFORMANCE:")
        tasks = ["mrpc", "sst2", "sentiment"]
        print(f"{'Task':<12} {'LoRA':<8} {'CURLoRA':<8} {'Difference':<12}")
        print("-" * 50)
        
        for task in tasks:
            lora_acc = lora["accuracies"][task]
            curlora_acc = curlora["accuracies"][task]
            diff = curlora_acc - lora_acc
            print(f"{task.upper():<12} {lora_acc:<8.4f} {curlora_acc:<8.4f} {diff:+.4f}")
        
        print()
        print("PERPLEXITY:")
        print(f"LoRA final perplexity:    {lora['final_perplexity']:.2f}")
        print(f"CURLoRA final perplexity: {curlora['final_perplexity']:.2f}")
        print()
        
        # Training time comparison
        total_lora_time = sum(lora["training_times"].values())
        total_curlora_time = sum(curlora["training_times"].values())
        print("TRAINING TIME:")
        print(f"LoRA total training time:    {total_lora_time:.2f}s")
        print(f"CURLoRA total training time: {total_curlora_time:.2f}s")
        print(f"Time difference: {total_curlora_time - total_lora_time:+.2f}s")
    
    def save_results(self, results):
        """Save results to JSON file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"curlora_comparison_{results['model_name'].replace('/', '_')}_{timestamp}.json"
        
        Path("results").mkdir(exist_ok=True)
        filepath = Path("results") / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Test CURLoRA vs LoRA")
    parser.add_argument("--model", default="gpt2-large", help="Model to test")
    parser.add_argument("--rank", type=int, default=8, help="Adaptation rank")
    parser.add_argument("--alpha", type=int, default=1, help="Adaptation alpha")
    parser.add_argument("--device", default="cpu", help="Device to use")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    print(f"Using device: {args.device}")
    
    # Run tests
    tester = ModelTester(args.model, args.device)
    results = tester.run_comparison(args.rank, args.alpha, not args.no_save)
    
    return results


if __name__ == "__main__":
    main()