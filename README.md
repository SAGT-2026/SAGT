# SAGT: Substructure-Aware Graph Tokenization for Large Language Models

A two-phase training framework that bridges graph neural networks and large language models through **graph motif quantization** and **topological Low-Rank Adaptation (TopoLoRA)**.

---

## Overview

SAGT introduces a pipeline for encoding graph-structured data into discrete motif tokens and injecting them into large language models (LLMs) via dynamically generated, graph-conditioned LoRA weights. The framework consists of two phases:

1. **Pretraining Phase** — A `GraphMotifQuantizer` learns compact, discrete representations of graph structures through vector quantization.
2. **Fine-tuning Phase** — The learned graph tokens are fused with a LLaMA-based LLM via `LlamatopoloraForCausalLM`, which injects graph topology information into all attention and MLP layers through dynamically generated low-rank weight updates.

An optional **Joint Training** mode allows the pretrained codebook and encoder to be fine-tuned simultaneously with the topological LoRA.

## Setup

```bash
# Create and activate the conda environment
conda env create -n SAGT python=3.11
conda activate SAGT

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Phase 1: Pretraining the Graph Motif Quantizer

Place your graph dataset file in the `Codebook/` directory, then run:

```bash
cd Codebook
python train.py 
  --dataset 
  --batch_size 
  --epochs 
  --lr 
  --device 
  --output_dir 
```


#### Training Losses

The `GraphMotifQuantizer` optimizes a composite loss:

$$\mathcal{L} = \mathcal{L}_\text{recon} + \lambda_\text{commit} \cdot \mathcal{L}_\text{commit} + \lambda_\text{vq} \cdot \mathcal{L}_\text{vq} + \lambda_\text{link} \cdot \mathcal{L}_\text{linkpred}$$

| Loss | Description |
|------|-------------|
| `reconstruction` | MSE (node features) + BCE (adjacency) |
| `commitment` | Encourages encoder outputs to commit to codebook entries |
| `vq` | Encourages codebook entries to track encoder outputs |
| `linkpred` | Cross-entropy link prediction from soft assignment matrices |

---

### Phase 2: Fine-tuning with Topological LoRA

Ensure a pretrained LLaMA-compatible model is available locally, then run:

```bash
cd Topological
python train.py \
  --model_path 
  --data_path 
  --output_dir 
  --per_device_train_batch_size 
  --num_train_epochs 
  --learning_rate 
  --graph_generator_lr 
  --A_matrix_lr 
  --freeze_llm 
  --topolora_rank 
  --topolora_type 
  --topolora_llm_dim 
  --topolora_llm_depth 
  --node_feat_dim 
  --graph_hidden_dim 
  --gnn_type 
  --gnn_layers 
```

#### LoRA Weight Generation Strategies

Three strategies are available via `--generation_method`:

**1. Direct** (`direct`, default)

A single linear projection from graph embeddings to all LoRA A matrices:
```
graph_embedding → Linear → A matrices (all layers × all types)
```

**2. Hypernetwork** (`hypernetwork`)

Factorized generation using a shared MLP conditioned on layer and type embeddings:
```
[graph_emb ‖ layer_emb ‖ type_emb] → MLP → U, V → A = U @ V
```
Additional arguments: `--hypernetwork_factor_k 8`, `--hypernetwork_hidden_dim 512`

**3. Prototype** (`prototype`)

Learns `K` prototype A matrices per type; the graph embedding produces soft mixing weights:
```
graph_embedding → MLP → softmax weights → weighted sum of prototypes → A
```
Additional arguments: `--prototype_K 4`, `--prototype_hidden_dim 256`

---

### Joint Training 

To fine-tune the pretrained codebook and encoder simultaneously with the topological LoRA:

```bash
cd Topological
python train.py \
  --model_path 
  --data_path
  --output_dir 
  --pretrained_codebook_path 
  --train_codebook_encoder 
  --codebook_lr
  --encoder_lr 
  --graph_generator_lr 
  --A_matrix_lr 
  --per_device_train_batch_size 
  --num_train_epochs 
```


## Data Format

### Pretraining

The pretraining script expects a serialized PyG `Dataset` saved as:

```python
torch.save(dataset, '.pt')
```

Internally, graphs are converted to padded dense tensors:
- `batch_x` — `[N, max_nodes, node_feat_dim]`
- `batch_adj` — `[N, max_nodes, max_nodes]` (symmetric, with self-loops)

### Fine-tuning

Conversation data should be a JSON file containing a list of conversation objects:

```json
[
  {
    "conversations": [
      {"from": "human", "value": "Here is the structure of a compound  <graph_start><graph><graph><graph><graph><graph><graph><graph><graph_end> . Could you assist me in predicting whether this compound exhibits benzodiazepine receptor activity?"},
      {"from": "gpt",   "value": "The aforementioned compound does not exhibit benzodiazepine receptor activity."}
    ]
  }
]
```

The `<graph>` tokens in the conversation are replaced at runtime with the corresponding quantized motif feature vectors.

---

## Inference: Getting Graph Motif Tokens

```python
import torch
from Codebook.train import create_model

# Load checkpoint
checkpoint = torch.load('final_model.pth', weights_only=False)
model = create_model(
    checkpoint['input_dim'],
    checkpoint['max_nodes'],
    checkpoint['config']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run tokenization
motif_results = model.get_motif_tokens(x, adj, batch_num_nodes)
tokens   = motif_results['tokens']            # [batch, num_clusters]  
mask     = motif_results['mask']              # [batch, num_clusters]  
features = motif_results['quantized_features']  # [batch, num_clusters, embedding_dim]
```

