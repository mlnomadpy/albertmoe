import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
from mteb import MTEB, get_tasks
from torch.optim import Adam
import transformers
from safetensors.torch import load_file
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AlbertConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from torch.optim.optimizer import Optimizer

# Set the transformers logger to warning level to suppress step-by-step loss prints
transformers.logging.set_verbosity_warning()



class ChillAdam(Optimizer):
    def __init__(self, params, min_lr=1e-4, max_lr=1.0, eps=1e-3, betas=(0.9, 0.999), weight_decay=0, lr= 1):
        """
        AdamChill: A hybrid optimizer combining Adam's momentum with ChillSGD's
        adaptive learning rate and gradient normalization.
        """
        if not 0.0 <= min_lr:
            raise ValueError(f"Invalid min_lr: {min_lr}")
        if not 0.0 <= max_lr:
            raise ValueError(f"Invalid max_lr: {max_lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid eps: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(min_lr=min_lr, max_lr=max_lr, eps=eps, betas=betas, weight_decay=weight_decay, lr=lr)
        super(ChillAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            min_lr, max_lr, eps, betas, weight_decay = group['min_lr'], group['max_lr'], group['eps'], group['betas'], group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamChill does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = betas
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                grad_norm = grad.norm(p=2).clamp(min=eps, max=100.)
                grad_normalized = grad / grad_norm

                if weight_decay != 0:
                    grad_normalized = grad_normalized.add(p, alpha=weight_decay)

                exp_avg.mul_(beta1).add_(grad_normalized, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_normalized, grad_normalized, value=1 - beta2)

                param_norm = p.norm(p=2).clamp(min=eps)
                lr = 1.0 / param_norm
                lr = lr.clamp(min=min_lr, max=max_lr)
                self.state[p]["lr"] = lr.item()

                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

# -----------------------------------------------------------------------------
# START: New Model Components (RoPE and MoE)
# -----------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x):
        seq_len = x.shape[2]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_emb = (q * cos) + (rotate_half(q) * sin)
    k_emb = (k * cos) + (rotate_half(k) * sin)
    return q_emb, k_emb

class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        return self.dropout(self.linear2(self.gelu(self.linear1(x))))

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        self.gate = nn.Linear(config.hidden_size, self.num_experts, bias=False)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        num_tokens = batch_size * seq_len
        x = x.view(-1, hidden_size)

        router_logits = self.gate(x)
        routing_weights, selected_experts = torch.topk(F.softmax(router_logits, dim=1, dtype=torch.float), self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        expert_counts = F.one_hot(selected_experts, num_classes=self.num_experts).sum(dim=1).sum(dim=0)
        f_i = expert_counts / num_tokens
        P_i = router_logits.softmax(dim=-1).mean(dim=0)
        aux_loss = self.num_experts * (f_i * P_i).sum()

        final_hidden_states = torch.zeros_like(x)

        flat_selected_experts = selected_experts.view(-1)

        for i, expert in enumerate(self.experts):
            expert_mask = (flat_selected_experts == i)
            flat_expert_indices = expert_mask.nonzero(as_tuple=True)[0]

            if flat_expert_indices.numel() == 0:
                continue

            expert_indices = flat_expert_indices // self.top_k
            tokens_for_expert = x[expert_indices]
            expert_output = expert(tokens_for_expert)
            weights_for_expert = routing_weights.view(-1)[expert_mask]

            final_hidden_states.index_add_(0, expert_indices, expert_output * weights_for_expert.unsqueeze(1).to(x.dtype))

        return final_hidden_states.view(batch_size, seq_len, hidden_size), aux_loss

# -----------------------------------------------------------------------------
# START: PyTorch ALBERT Model Implementation (Updated)
# -----------------------------------------------------------------------------

class AlbertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.segment_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        token_embeds = self.token_embeddings(input_ids)
        segment_embeds = self.segment_embeddings(token_type_ids)

        embeddings = token_embeds + segment_embeds
        projected_embeddings = self.embedding_hidden_mapping_in(embeddings)
        normalized_embeddings = self.norm(projected_embeddings)
        final_embeddings = self.dropout(normalized_embeddings)
        return final_embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        assert config.hidden_size % self.num_heads == 0, "Hidden size must be divisible by num_heads"
        self.qkv_layer = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, hidden_size = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        cos, sin = self.rotary_emb(v)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if attention_mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2) == 0, mask_value)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, v)
        context = context.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
        output = self.output_proj(context)
        return output

class AlbertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.ffn = MoE(config)
        self.norm1 = nn.RMSNorm(config.hidden_size)
        self.norm2 = nn.RMSNorm(config.hidden_size)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x, attention_mask=None):
        attn_output = self.attention(x, attention_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output, aux_loss = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x, aux_loss

class ALBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.shared_layer = AlbertLayer(config)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.embeddings(input_ids, token_type_ids)
        total_aux_loss = 0.0
        for _ in range(self.config.num_hidden_layers):
            x, aux_loss = self.shared_layer(x, attention_mask)
            total_aux_loss += aux_loss
        return x, total_aux_loss / self.config.num_hidden_layers

class AlbertForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.albert = ALBERT(config)
        self.mlm_head_transform = nn.Linear(config.hidden_size, config.embedding_size, bias=False)
        self.gelu = nn.GELU()
        self.norm = nn.RMSNorm(config.embedding_size)
        self.mlm_head_decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
        self.config = config

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        hidden_states, aux_loss = self.albert(input_ids, attention_mask, token_type_ids)
        transformed_states = self.norm(self.gelu(self.mlm_head_transform(hidden_states)))
        logits = self.mlm_head_decoder(transformed_states)

        loss, mlm_loss = None, None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            mlm_loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = mlm_loss + self.config.aux_loss_coefficient * aux_loss
            loss = total_loss

        return {"loss": loss, "mlm_loss": mlm_loss, "logits": logits, "aux_loss": aux_loss}

# -----------------------------------------------------------------------------
# Wrapper for MTEB Evaluation
# -----------------------------------------------------------------------------

class SentenceAlbert(nn.Module):
    def __init__(self, config, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.albert = ALBERT(config)

        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
        self.albert.load_state_dict(state_dict)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.albert.to(self.device)
        self.albert.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # ALBERT now returns a tuple
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=32, **kwargs):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128).to(self.device)
            model_output = self.albert(**inputs)
            sentence_embedding = self.mean_pooling(model_output, inputs["attention_mask"])
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=1)
            all_embeddings.append(sentence_embedding.cpu())
        return torch.cat(all_embeddings, dim=0).numpy()

# -----------------------------------------------------------------------------
# Main Training and Evaluation Script
# -----------------------------------------------------------------------------

def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Pre-train or evaluate a custom ALBERT model.")

    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument("--mode", type=str, default="all", choices=["pretrain", "evaluate", "all"])
    mode_group.add_argument("--model_path", type=str, default="./albert_from_scratch_output")

    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument("--embedding_size", type=int, default=128)
    model_group.add_argument("--hidden_size", type=int, default=768)
    model_group.add_argument("--num_hidden_layers", type=int, default=12)
    model_group.add_argument("--num_attention_heads", type=int, default=12)
    model_group.add_argument("--intermediate_size", type=int, default=3072)
    model_group.add_argument("--num_experts", type=int, default=8, help="Number of experts in MoE layer")
    model_group.add_argument("--top_k_experts", type=int, default=2, help="Number of experts to use per token")

    dataset_group = parser.add_argument_group('Dataset Configuration')
    dataset_group.add_argument("--dataset", type=str, default="wikitext")
    dataset_group.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    dataset_group.add_argument("--dataset_split", type=str, default="train")
    dataset_group.add_argument("--text_column", type=str, default="text")
    dataset_group.add_argument("--streaming", action="store_true", default=False)
    dataset_group.add_argument("--max_samples", type=int, default=100000)

    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument("--num_epochs", type=int, default=1)
    train_group.add_argument("--batch_size", type=int, default=64)
    train_group.add_argument("--max_length", type=int, default=128)
    train_group.add_argument("--min_lr", type=float, default=1e-4)
    train_group.add_argument("--max_lr", type=float, default=1.0)
    train_group.add_argument("--aux_loss_coefficient", type=float, default=0.001, help="Coefficient for MoE auxiliary loss")
    train_group.add_argument("--l1_lambda", type=float, default=1e-5, help="L1 regularization strength (Lasso)")


    eval_group = parser.add_argument_group('Evaluation Configuration')
    eval_group.add_argument("--eval_tasks", type=str, default="comprehensive", choices=["basic", "comprehensive", "custom"])
    eval_group.add_argument("--custom_tasks", type=str, nargs="+", default=["STSBenchmark"])

    args = parser.parse_args()

    # --- Environment Setup ---
    os.environ["WANDB_PROJECT"] = "albert_iclr"
    output_dir = "./albert_from_scratch_output"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    albert_config = AlbertConfig(
        vocab_size=tokenizer.vocab_size, embedding_size=args.embedding_size,
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads, intermediate_size=args.intermediate_size,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=args.max_length,
        type_vocab_size=2, pad_token_id=tokenizer.pad_token_id,
        num_experts=args.num_experts, top_k_experts=args.top_k_experts,
        aux_loss_coefficient=args.aux_loss_coefficient,
        l1_lambda=args.l1_lambda
    )

    # --- PRETRAINING SECTION ---
    if args.mode in ["pretrain", "all"]:
        print("\n" + "="*60 + "\nSTARTING PRE-TRAINING MODE\n" + "="*60)

        model = AlbertForMaskedLM(config=albert_config).to(device)
        print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

        # --- Dataset ---
        try:
            config_to_use = None if args.dataset_config.lower() in ['none', 'null', ''] else args.dataset_config
            dataset = load_dataset(args.dataset, config_to_use, split=args.dataset_split, streaming=args.streaming)
        except Exception as e:
            print(f"✗ Error loading dataset: {e}. Falling back to wikitext.")
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=args.streaming)
            args.text_column = "text"

        def tokenize_function(examples):
            return tokenizer(examples[args.text_column], truncation=True, padding="max_length", max_length=args.max_length)

        if args.streaming:
            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=list(dataset.features.keys()))
            if args.max_samples: tokenized_dataset = tokenized_dataset.take(args.max_samples)
        else:
            if args.max_samples: dataset = dataset.select(range(min(args.max_samples, len(dataset))))
            tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=dataset.column_names)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
        num_workers = max(0, os.cpu_count() - 2)
        dataloader = DataLoader(
            tokenized_dataset, batch_size=args.batch_size, collate_fn=data_collator,
            num_workers=num_workers, pin_memory=device.type == "cuda", prefetch_factor=2 if num_workers > 0 else None
        )

        # --- Training ---
        optimizer = ChillAdam(model.parameters(), min_lr=args.min_lr, max_lr=args.max_lr)
        run_name = f"albert-pretrain-{args.dataset.replace('/', '-')}"
        wandb.init(project=os.environ["WANDB_PROJECT"], name=run_name, config=vars(args))
        wandb.watch(model, log="all", log_freq=100)

        global_step = 0
        model.train()
        for epoch in range(args.num_epochs):
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                outputs = model(**batch)
                loss = outputs['loss']
                mlm_loss = outputs['mlm_loss']
                aux_loss = outputs['aux_loss']

                # --- L1 Regularization (Lasso) ---
                l1_penalty = 0.0
                if model.config.l1_lambda > 0:
                    for param in model.parameters():
                        l1_penalty += torch.abs(param).sum()
                    loss += model.config.l1_lambda * l1_penalty
                # --- End L1 Regularization ---

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                lrs = [s['lr'] for s in optimizer.state.values() if 'lr' in s]
                log_data = {
                    "train/total_loss": loss.item(),
                    "train/mlm_loss": mlm_loss.item(),
                    "train/aux_loss": aux_loss.item(),
                    "train/l1_penalty": l1_penalty.item() if isinstance(l1_penalty, torch.Tensor) else l1_penalty,
                    "lr/mean": sum(lrs)/len(lrs) if lrs else 0
                }
                wandb.log(log_data, step=global_step)
                progress_bar.set_postfix(total_loss=loss.item(), mlm_loss=mlm_loss.item(), aux_loss=aux_loss.item(), l1_penalty=log_data['train/l1_penalty'])

                if args.streaming and args.max_samples and global_step * args.batch_size >= args.max_samples: break
            if args.streaming and args.max_samples and global_step * args.batch_size >= args.max_samples: break

        print("\nPRE-TRAINING COMPLETE. Saving model...")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.albert.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(output_dir)

    # --- EVALUATION SECTION ---
    if args.mode in ["evaluate", "all"]:
        print(f"\n--- Starting MTEB Evaluation on: {args.model_path} ---")
        mteb_model = SentenceAlbert(config=albert_config, model_path=args.model_path)

        if args.eval_tasks == "basic": task_list = ["STSBenchmark"]
        elif args.eval_tasks == "comprehensive": task_list = [t.metadata.name for t in get_tasks()]
        else: task_list = args.custom_tasks

        evaluation = MTEB(tasks=task_list)
        results = evaluation.run(mteb_model, output_folder=f"results/{os.path.basename(args.model_path)}_{args.eval_tasks}")

        if wandb.run is None: wandb.init(project=os.environ["WANDB_PROJECT"], name=f"evaluate-{os.path.basename(args.model_path)}", config=vars(args))

        all_logged_metrics = {}
        for task_result in results:
            task_name = task_result.task_name
            for split_name, scores in task_result.scores.items():
                if isinstance(scores, list):
                    for i, score_dict in enumerate(scores):
                        for metric_name, value in score_dict.items():
                             if isinstance(value, (int, float)):
                                key = f"eval/{task_name}/{split_name}_{i}/{metric_name}"
                                all_logged_metrics[key] = value
                elif isinstance(scores, dict):
                    if any(isinstance(v, dict) for v in scores.values()):
                        for main_metric, metric_dict in scores.items():
                            for metric_name, value in metric_dict.items():
                                if isinstance(value, (int, float)):
                                    all_logged_metrics[f"eval/{task_name}/{split_name}/{main_metric}_{metric_name}"] = value
                    else:
                        for metric_name, value in scores.items():
                            if isinstance(value, (int, float)):
                                all_logged_metrics[f"eval/{task_name}/{split_name}/{metric_name}"] = value

        if all_logged_metrics:
            wandb.log(all_logged_metrics)
            print(f"✅ Logged {len(all_logged_metrics)} metrics to W&B.")

    if wandb.run is not None:
        wandb.finish()

    print("\nSCRIPT EXECUTION COMPLETE")

if __name__ == "__main__":
    main()

