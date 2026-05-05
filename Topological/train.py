import os
import copy
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
from transformers import AutoModel,AutoTokenizer

import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from weight_generate import get_graph_lora_generater
import graph_tokenize
from Topological.llama_lora import LlamatopoloraForCausalLM

from constants import IGNORE_INDEX
from Topological.sagt_trainer import SAGTTrainer

local_rank = None


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        
        output_embeddings_layer = model.get_output_embeddings()
        if output_embeddings_layer is not None:
            output_embeddings = output_embeddings_layer.weight.data
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings[-num_new_tokens:] = output_embeddings_avg



def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass(frozen=False)
class ModelArguments:
    model_path: Optional[str] = field(default="../Model/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    
    topolora_dim: Optional[int] = field(default=512)
    topolora_depth: Optional[int] = field(default=6)
    node_feat_dim: Optional[int] = field(default=512)
    graph_hidden_dim: Optional[int] = field(default=256)
    topolora_pos_num: Optional[int] = field(default=32)
    topolora_llm_dim: Optional[int] = field(default=4096)
    topolora_llm_depth: Optional[int] = field(default=32)
    topolora_rank: Optional[int] = field(default=64)
    topolora_type: Optional[str] = field(default='qkvo')
    topolora_alpha: Optional[int] = field(default=None)
    weights_sep: Optional[bool] = field(default=True)
    skip_layers: Optional[int] = field(default=1)
    gnn_type: Optional[str] = field(default='GAT')
    gnn_layers: Optional[int] = field(default=3)

    generation_method: Optional[str] = field(default='direct',
        metadata={"help": "A-matrix generation method: 'direct', 'hypernetwork', or 'prototype'"})
    hypernetwork_factor_k: Optional[int] = field(default=8,
        metadata={"help": "Factorization rank k for hypernetwork approach (A = U@V)"})
    hypernetwork_hidden_dim: Optional[int] = field(default=512,
        metadata={"help": "Hidden dimension of the shared hypernetwork MLP"})
    prototype_K: Optional[int] = field(default=4,
        metadata={"help": "Number of prototype A matrices per type for prototype mixing"})
    prototype_hidden_dim: Optional[int] = field(default=256,
        metadata={"help": "Hidden dimension of the mixing weight MLP"})

    graph_data_path: Optional[str] = field(default=None)
    use_pretrained_graph_tokenizer: Optional[bool] = field(default=True)
    
    # Parameters for codebook and encoder training
    pretrained_codebook_path: Optional[str] = field(default=None)
    train_codebook_encoder: Optional[bool] = field(default=False)
    codebook_lr: Optional[float] = field(default=1e-5)
    encoder_lr: Optional[float] = field(default=1e-5)


@dataclass(frozen=False)
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    graph_folder: Optional[str] = field(default=None)
    max_graph_size: Optional[int] = field(default=100)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=512)
    
    
    freeze_llm: bool = field(default=True)
    graph_generator_lr: Optional[float] = field(default=1e-4)
    A_matrix_lr: Optional[float] = field(default=1e-3)
    
    # Learning rates for codebook and encoder  
    codebook_lr: Optional[float] = field(default=1e-5)
    encoder_lr: Optional[float] = field(default=1e-5)
    
    bits: int = field(default=16)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf8")


class GraphLoRADataset(Dataset):
    
    def __init__(self, data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments, model_args: ModelArguments,
                 pretrained_model=None):
        super().__init__()
        
        self.list_data_dict = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_args = model_args
        self.pretrained_model = pretrained_model
        
        self._init_graph_data()
         
    def _init_graph_data(self):
        if self.model_args.use_pretrained_graph_tokenizer:
            if self.pretrained_model is not None:
                # Use provided pretrained model for tokenization
                result = self._tokenize_with_pretrained_model()
            else:
                # Use default loading method
                result = graph_tokenize.load_pretrained_model_and_tokenize() 
            self.adjacency_matrices = result['adj_matrix']
            self.node_features = result['motif_features']
        else:
            raise NotImplementedError("Custom graph loading not implemented yet")
            
        if len(self.adjacency_matrices) > 0:
            sample_nodes = self.adjacency_matrices[0].shape[0]
            sample_feat_dim = self.node_features[0].shape[1]
            self.position_encoding = graph_tokenize.generate_sinusoidal_position_encoding(
                sample_nodes, sample_feat_dim
            )
        for i in range(len(self.node_features)):
            self.node_features[i] = self.node_features[i]+ self.position_encoding
    
    def _tokenize_with_pretrained_model(self):
        """Use the loaded pretrained model for graph tokenization"""
        import Codebook.dataloader as dataloader
        
        try:
            dataset_origin = torch.load('sider_hepatobiliary_dataset.pt', weights_only=False)
            batch_x, batch_adj, batch_num_nodes, _, max_nodes = dataloader.parse_the_dataset_to_matrices(dataset_here=dataset_origin)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tokenization_results = graph_tokenize.tokenize_graphs(
                self.pretrained_model, batch_x, batch_adj, batch_num_nodes, device
            )
            
            cluster_adj = tokenization_results['cluster_adj']
            batch_size = cluster_adj.shape[0]
            processed_adj_matrices = []
            
            for i in range(batch_size):
                new_adj_matrix = (cluster_adj[i] > 1).float()
                new_adj_matrix = graph_tokenize.remove_diagonal(new_adj_matrix)
                processed_adj_matrices.append(new_adj_matrix)
            
            normalized_adj_batch = torch.stack(processed_adj_matrices, dim=0)
            
            return {
                **tokenization_results,
                'motif_features': tokenization_results['quantized_features'], 
                'adj_matrix': normalized_adj_batch
            }
        except Exception as e:
            print(f"Warning: Failed to tokenize with pretrained model: {e}")
            print("Falling back to default tokenization method")
            return graph_tokenize.load_pretrained_model_and_tokenize()
        
            
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(sources, dict):
            sources = [sources]
        
        conv = self._get_conversation_template() 
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        
        conversations = []
        for source in sources:
            if roles[source['conversations'][0]["from"]] != conv.roles[0]:
                source['conversations'] = source['conversations'][1:]
            
            conv.messages = []
            for j, sentence in enumerate(source['conversations']):
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]    
        labels = input_ids.clone()
        sep = conv.sep + conv.roles[1] + ": "
        conversation = conversations[0]
        total_len = int(labels.ne(self.tokenizer.pad_token_id).sum())
        rounds = [conversation] 
        cur_len = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep 
            
            round_len = len(self.tokenizer(rou).input_ids)
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2
            
            labels[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
                 
        labels[cur_len:] = IGNORE_INDEX
        if cur_len < self.tokenizer.model_max_length:
            if cur_len != total_len:
                print("cur_len != total_len, setting labels to IGNORE_INDEX")
                labels[:] = IGNORE_INDEX
        
        graph_idx = i % len(self.adjacency_matrices)
        adjacency_matrix = self.adjacency_matrices[graph_idx].clone().detach().to(dtype=torch.float32,device='cpu')
        node_features = (self.node_features[graph_idx] + self.position_encoding).clone().detach().to(dtype=torch.float32,device='cpu')
        motif_features = self.node_features[graph_idx].clone().detach().to(dtype=torch.float32,device='cpu')
        return {
            'input_ids': input_ids,
            'labels': labels,
            'adjacency_matrix': adjacency_matrix,
            'node_features': node_features,
            'motif_features': motif_features
        }
        
        
    def _get_conversation_template(self):
        class SimpleConversation:
            def __init__(self):
                self.roles = ["Human", "Assistant"]
                self.sep = "\n"
                self.sep2 = "\n"
                self.messages = []
                
            def append_message(self, role, message):
                self.messages.append([role, message])
                
            def get_prompt(self):
                ret = ""
                for i, (role, message) in enumerate(self.messages):
                    if i == 0:
                        ret += role + ": " + message + self.sep
                    else:
                        ret += role + ": " + message + self.sep2
                return ret
            
        return SimpleConversation()

@dataclass
class GraphLoRADataCollator:
    
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance['input_ids'] for instance in instances]
        labels = [instance['labels'] for instance in instances]
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        
        adjacency_matrices = [instance['adjacency_matrix'] for instance in instances]
        node_features = [instance['node_features'] for instance in instances]
        motif_features = [instance['motif_features'] for instance in instances] 
        
        
        batch = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
            'adjacency_matrices': adjacency_matrices,
            'node_features': node_features,
            'motif_features': motif_features, 
        }
        return batch

class GraphLoRATrainer(SAGTTrainer):
    
    def __init__(self, graph_lora_generator=None, pretrained_model=None, model_args=None, **kwargs):
        super().__init__(**kwargs)
        self.pretrained_model = pretrained_model
        self.model_args = model_args
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Compute the main LLM loss
        outputs = model(**inputs)
        main_loss = outputs.loss
        
        # If we have a pretrained model and it's trainable, compute additional losses
        additional_loss = 0.0
        if (self.pretrained_model is not None and 
            self.model_args is not None and 
            self.model_args.train_codebook_encoder and
            'adjacency_matrices' in inputs and 
            'motif_features' in inputs):
            
            try:
                # Extract graph data from inputs
                adjacency_matrices = inputs['adjacency_matrices']
                motif_features = inputs['motif_features']
                
                # Process a subset of the batch to avoid memory issues
                batch_size = min(len(adjacency_matrices), 4)  # Limit batch size for graph processing
                
                total_graph_loss = 0.0
                valid_samples = 0
                
                for i in range(batch_size):
                    try:
                        adj_matrix = adjacency_matrices[i].to(main_loss.device)
                        node_features = motif_features[i].to(main_loss.device)
                        
                        # Add batch dimension if needed
                        if adj_matrix.dim() == 2:
                            adj_matrix = adj_matrix.unsqueeze(0)
                        if node_features.dim() == 2:
                            node_features = node_features.unsqueeze(0)
                        
                        # Get number of nodes
                        num_nodes = adj_matrix.shape[-1]
                        batch_num_nodes = [num_nodes]
                        
                        # Forward pass through pretrained model
                        results = self.pretrained_model(
                            node_features, adj_matrix, batch_num_nodes, return_intermediate=True
                        )
                        
                        # Compute losses
                        graph_losses = self.pretrained_model.compute_losses(results)
                        
                        # Weighted combination of losses
                        sample_loss = (
                            graph_losses['reconstruction'] * 0.5 +
                            graph_losses['commitment'] * 0.3 +
                            graph_losses['vq'] * 0.2
                        )
                        
                        total_graph_loss += sample_loss
                        valid_samples += 1
                        
                    except Exception as e:
                        # Skip problematic samples
                        print(f"Warning: Skipping graph sample {i} due to error: {e}")
                        continue
                
                if valid_samples > 0:
                    additional_loss = total_graph_loss / valid_samples * 0.1  # Scale down the graph loss
                
            except Exception as e:
                print(f"Warning: Failed to compute graph losses: {e}")
                additional_loss = 0.0
        
        # Combine losses
        total_loss = main_loss + additional_loss
        
        if return_outputs:
            # Add loss information to outputs if needed
            if hasattr(outputs, 'loss'):
                outputs.loss = total_loss
            return total_loss, outputs
        else:
            return total_loss
    
def make_graph_lora_data_module(tokenizer, data_args, model_args, pretrained_model=None) -> Dict:
    
    train_dataset = GraphLoRADataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        pretrained_model=pretrained_model
    )
    
    data_collator = GraphLoRADataCollator(tokenizer=tokenizer)
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator
    )


def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    
    compute_dtype = (
        torch.float16 if training_args.fp16 
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    print(f"Using compute dtype: {compute_dtype}")
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type
            )
        ))
    
    if model_args.topolora_alpha is None:
        model_args.topolora_alpha = model_args.topolora_rank

    # Load pretrained codebook and encoder if specified
    pretrained_model = None
    if model_args.train_codebook_encoder and model_args.pretrained_codebook_path:
        rank0_print(f"Loading pretrained model from {model_args.pretrained_codebook_path}")
        try:
            import Codebook.train as codebook_train
            
            # Load the model configuration and state
            checkpoint = torch.load(model_args.pretrained_codebook_path, map_location=device, weights_only=False)
            config = checkpoint.get('config', {})
            input_dim = checkpoint.get('input_dim', 30)
            max_nodes = checkpoint.get('max_nodes', 40)
            
            # Create and load the pretrained model
            pretrained_model = codebook_train.create_model(input_dim, max_nodes, config)
            pretrained_model.load_state_dict(checkpoint['model_state_dict'])
            pretrained_model = pretrained_model.to(device)
            
            # Set trainable parameters for codebook and encoder
            if model_args.train_codebook_encoder:
                for param in pretrained_model.parameters():
                    param.requires_grad = True
                rank0_print("Pretrained model loaded and set to trainable")
            else:
                for param in pretrained_model.parameters():
                    param.requires_grad = False
                rank0_print("Pretrained model loaded and frozen")
                
        except Exception as e:
            rank0_print(f"Failed to load pretrained model: {e}")
            rank0_print("Continuing without pretrained model")
            pretrained_model = None

    graph_lora_generator = get_graph_lora_generater(model_args)
    
    model = LlamatopoloraForCausalLM.from_pretrained(
        model_args.model_path,
        cache_dir=training_args.cache_dir,
        local_files_only=True,
        graph_lora_generator=graph_lora_generator, 
        **bnb_model_from_pretrained_args
    )
    
    model.config.use_cache = False
    
    if training_args.freeze_llm:
        for name, param in model.named_parameters():
            if not name.startswith("graph_lora_generator"):
                param.requires_grad = False
                     
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = compute_dtype
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    special_token_dict = {
        'additional_special_tokens': [
            '<graph>',
            '<graph_start>',
            '<graph_end>',
        ]
    }
    
    if model_args.version == "v0":
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_token_dict,
            tokenizer=tokenizer,
            model=model
        )

    print("\nThe new additional tokens:")
    for token in ['<graph>', '<graph_start>', '<graph_end>']:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token}: ID {token_id}")
        
    
    data_module = make_graph_lora_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        model_args=model_args,
        pretrained_model=pretrained_model
    )

    total_params = sum(p.numel() for p in graph_lora_generator.parameters() if p.requires_grad)
    rank0_print(f"Graph LoRA Generator learnable parameters: {total_params/1e6:.2f}M")
    
    if pretrained_model is not None:
        pretrained_params = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
        rank0_print(f"Pretrained model learnable parameters: {pretrained_params/1e6:.2f}M")
    
    trainer = GraphLoRATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        pretrained_model=pretrained_model,
        model_args=model_args,
        **data_module
    )
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in graph_lora_generator.named_parameters() if "As" not in n and p.requires_grad],
            "lr": training_args.graph_generator_lr or training_args.learning_rate,
        },
        {
            "params": [p for n, p in graph_lora_generator.named_parameters() if "As" in n and p.requires_grad],
            "lr": training_args.A_matrix_lr or training_args.learning_rate * 10,
        }
    ]
    
    # Add pretrained model parameters to optimizer if training them
    if pretrained_model is not None and model_args.train_codebook_encoder:
        # Add codebook parameters
        codebook_params = []
        encoder_params = []
        other_params = []
        
        for name, param in pretrained_model.named_parameters():
            if param.requires_grad:
                if 'codebook' in name.lower():
                    codebook_params.append(param)
                elif 'encoder' in name.lower():
                    encoder_params.append(param)
                else:
                    other_params.append(param)
        
        if codebook_params:
            optimizer_grouped_parameters.append({
                "params": codebook_params,
                "lr": training_args.codebook_lr or training_args.learning_rate * 0.1,
            })
            rank0_print(f"Added {len(codebook_params)} codebook parameters to optimizer")
        
        if encoder_params:
            optimizer_grouped_parameters.append({
                "params": encoder_params,
                "lr": training_args.encoder_lr or training_args.learning_rate * 0.1,
            })
            rank0_print(f"Added {len(encoder_params)} encoder parameters to optimizer")
        
        if other_params:
            optimizer_grouped_parameters.append({
                "params": other_params,
                "lr": training_args.learning_rate * 0.1,
            })
            rank0_print(f"Added {len(other_params)} other pretrained model parameters to optimizer")
    
    # 设置优化器
    from torch.optim import AdamW
    trainer.optimizer = AdamW(optimizer_grouped_parameters, lr=5e-4)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        graph_generator_path = os.path.join(training_args.output_dir, 'graph_lora_generator.bin')
        torch.save(graph_lora_generator.state_dict(), graph_generator_path)
        rank0_print(f"Graph LoRA generator saved to {graph_generator_path}")
        
        # Save pretrained model if it was trained
        if pretrained_model is not None and model_args.train_codebook_encoder:
            pretrained_model_path = os.path.join(training_args.output_dir, 'updated_pretrained_model.pth')
            torch.save({
                'model_state_dict': pretrained_model.state_dict(),
                'config': checkpoint.get('config', {}),
                'input_dim': checkpoint.get('input_dim', 30),
                'max_nodes': checkpoint.get('max_nodes', 40)
            }, pretrained_model_path)
            rank0_print(f"Updated pretrained model saved to {pretrained_model_path}")

if __name__ == "__main__":
    train()
