import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import argparse
from datetime import datetime
import seaborn as sns
from sklearn.decomposition import PCA
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
import dataloader
from decoder import GraphMotifQuantizer, train_graph_motif_quantizer


def create_data_loaders(dataset_name=None, batch_size=32, train_ratio=0.8):
    print(f"Loading dataset: {dataset_name}")
    
    dataset_origin = torch.load('merged_dataset.pt', weights_only=False)
    dataset = dataloader.load_the_dataset(batch_size=batch_size, shuffle=True, num_workers=0, dataset=dataset_origin)
    batch_x, batch_adj, batch_num_nodes, _, max_nodes = dataloader.parse_the_dataset_to_matrices(dataset_here=dataset_origin)

    print(f"Dataset size: {len(batch_x)}")
    print(f"Node feature dimension: {batch_x.shape[2]}")
    print(f"Maximum nodes: {max_nodes}")
    
    total_size = len(batch_x)
    train_size = int(train_ratio * total_size)
    
    train_x = batch_x[:train_size]
    train_adj = batch_adj[:train_size]
    train_nodes = batch_num_nodes[:train_size]
    
    val_x = batch_x[train_size:]
    val_adj = batch_adj[train_size:]
    val_nodes = batch_num_nodes[train_size:]
    
    print(f"Training set size: {len(train_x)}")
    print(f"Validation set size: {len(val_x)}")
    
    train_dataset = TensorDataset(train_x, train_adj, torch.tensor(train_nodes))
    val_dataset = TensorDataset(val_x, val_adj, torch.tensor(val_nodes))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
   
    return train_loader, val_loader, batch_x.shape[2], max_nodes


def create_model(input_dim, max_nodes, config):
    model = GraphMotifQuantizer(
        max_num_nodes=max_nodes,
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        embedding_dim=config['embedding_dim'],
        num_layers=config['num_layers'],
        assign_hidden_dim=config['assign_hidden_dim'],
        assign_ratio=config['assign_ratio'],
        num_pooling=config['num_pooling'],
        
        codebook_size=config['codebook_size'],
        codebook_type=config['codebook_type'],
        num_codebooks=config['num_codebooks'],
        decay=config['decay'],
        threshold_ema_dead_code=config['threshold_ema_dead_code'],
        sample_codebook_temp=config['sample_codebook_temp'],
        gaussian_delta=config['gaussian_delta'],
        
        decoder_hidden_dim=config.get('decoder_hidden_dim', config['hidden_dim']),
        decoder_num_layers=config.get('decoder_num_layers', config['num_layers']),
        unpooling_method=config['unpooling_method'],
        
        commitment_weight=config['commitment_weight'],
        reconstruction_weight=config['reconstruction_weight'],
        structure_weight=config['structure_weight'],
        vq_weight=config['vq_weight'],
        linkpred_weight=config['linkpred_weight'],
        linkpred=config['linkpred'],
        dropout=config['dropout'],
        bn=config['bn']
    )
    
    return model


def test_model(model, test_loader, device):
    print("\n=== Testing trained model ===")
    model.eval()
    
    total_losses = {key: 0.0 for key in ['total', 'reconstruction', 'commitment', 'vq']}
    num_batches = 0
    
    sample_results = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            x, adj, batch_num_nodes = batch_data
            x = x.to(device)
            adj = adj.to(device)
            batch_num_nodes = batch_num_nodes.tolist()
            
            results = model(x, adj, batch_num_nodes, return_intermediate=True)
            losses = model.compute_losses(results)
            
            for key, value in losses.items():
                if key in total_losses:
                    total_losses[key] += value.item()
            num_batches += 1
            
            if batch_idx < 3:
                sample_results.append({
                    'motif_tokens': results['quantized_indices'],
                    'cluster_mask': results['cluster_mask'],
                    'batch_num_nodes': batch_num_nodes
                })
    
    for key in total_losses:
        total_losses[key] /= num_batches
    
    print(f"Test results:")
    print(f"  Average total loss: {total_losses['total']:.6f}")
    print(f"  Reconstruction loss: {total_losses['reconstruction']:.6f}")
    print(f"  Commitment loss: {total_losses['commitment']:.6f}")
    print(f"  VQ loss: {total_losses['vq']:.6f}")
    
    print(f"\n=== Motif Token Analysis ===")
    for i, result in enumerate(sample_results):
        tokens = result['motif_tokens']
        mask = result['cluster_mask']
        nodes = result['batch_num_nodes']
        
        print(f"\nBatch {i+1}:")
        for j in range(min(3, tokens.shape[0])):
            valid_tokens = tokens[j][mask[j]]
            print(f"  Graph {j+1} (nodes: {nodes[j]}): {valid_tokens.tolist()}")
            if len(valid_tokens) > 0:
                unique_tokens, counts = torch.unique(valid_tokens, return_counts=True)
                print(f"    Used tokens: {len(unique_tokens)}")
                print(f"    Token distribution: {dict(zip(unique_tokens.tolist(), counts.tolist()))}")
    
    return total_losses, sample_results


def visualize_training_progress(log_dir):
    print(f"\nTraining logs saved in: {log_dir}")
    print("Use the following command to view training progress:")
    print(f"tensorboard --logdir {log_dir}")


def save_initial_codebooks(model, save_dir):
    initial_codebooks = {}
    
    if hasattr(model.codebook, 'embed'):
        print("  - Found direct embed attribute")
        initial_codebooks['codebook'] = model.codebook.embed.clone().detach()
        print(f"    Shape: {initial_codebooks['codebook'].shape}")
            
    torch.save(initial_codebooks, os.path.join(save_dir, 'initial_codebooks.pth'))
    print(f"Initial codebooks saved: {len(initial_codebooks)} codebooks")
    
    return initial_codebooks


def extract_final_codebooks(model):
    final_codebooks = {}
    
    if hasattr(model.codebook, 'embed'):
        print("  - Found direct embed attribute")
        final_codebooks['codebook'] = model.codebook.embed.clone().detach()
        print(f"    Shape: {final_codebooks['codebook'].shape}")
    
    return final_codebooks


def compute_codebook_differences(initial_codebooks, final_codebooks):
    differences = {}
    for key in initial_codebooks.keys():
        if key in final_codebooks:
            initial = initial_codebooks[key]
            final = final_codebooks[key]
            
            l2_diff = torch.norm(final - initial, p=2, dim=-1)
            cosine_sim = torch.cosine_similarity(initial, final, dim=-1)
            
            differences[key] = {
                'l2_distances': l2_diff,
                'cosine_similarities': cosine_sim,
                'mean_l2_distance': l2_diff.mean().item(),
                'std_l2_distance': l2_diff.std().item(),
                'mean_cosine_similarity': cosine_sim.mean().item(),
                'std_cosine_similarity': cosine_sim.std().item(),
                'max_l2_distance': l2_diff.max().item(),
                'min_l2_distance': l2_diff.min().item(),
                'initial_shape': initial.shape,
                'final_shape': final.shape
            }
    
    return differences


def visualize_codebook_changes(initial_codebooks, final_codebooks, differences, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for key in initial_codebooks.keys():
        if key not in final_codebooks:
            continue
         
        initial_tensor = initial_codebooks[key]
        final_tensor = final_codebooks[key]
        
        if initial_tensor.dim() == 3 and initial_tensor.shape[0] == 1:
            initial_tensor = initial_tensor.squeeze(0)
        if final_tensor.dim() == 3 and final_tensor.shape[0] == 1:
            final_tensor = final_tensor.squeeze(0)
        
        initial = initial_tensor.cpu().numpy()
        final = final_tensor.cpu().numpy()
        diff_data = differences[key]
        
        l2_distances = diff_data['l2_distances'].cpu().numpy().flatten()
        cosine_similarities = diff_data['cosine_similarities'].cpu().numpy().flatten()
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 3, 1)
        plt.hist(l2_distances, bins=30, alpha=0.7, color='#5BC2D9')
        plt.xlabel('L2 Distance')
        plt.ylabel('Frequency')
        plt.title(f'{key}: L2 Distance Distribution')
        plt.axvline(diff_data['mean_l2_distance'], color='red', linestyle='--', 
                   label=f'Mean: {diff_data["mean_l2_distance"]:.4f}')
        plt.legend()
        
        plt.subplot(2, 3, 2)
        plt.hist(cosine_similarities, bins=30, alpha=0.7, color='#17E6B2')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title(f'{key}: Cosine Similarity Distribution')
        plt.axvline(diff_data['mean_cosine_similarity'], color='red', linestyle='--',
                   label=f'Mean: {diff_data["mean_cosine_similarity"]:.4f}')
        plt.legend()
        
        plt.subplot(2, 3, 3)
        num_codes = min(50, initial.shape[0])
        l2_matrix = l2_distances[:num_codes].reshape(-1, 1)
        sns.heatmap(l2_matrix, cmap='viridis', cbar=True)
        plt.title(f'{key}: L2 Distances (first {num_codes} codes)')
        plt.ylabel('Code Index')
        
        if initial.shape[1] > 2:
            plt.subplot(2, 3, 4)
            pca = PCA(n_components=2)
            
            combined = np.vstack([initial, final])
            pca_result = pca.fit_transform(combined)
            
            initial_pca = pca_result[:len(initial)]
            final_pca = pca_result[len(initial):]
            
            plt.scatter(initial_pca[:, 0], initial_pca[:, 1], alpha=0.6, 
                       label='Initial', color='red', s=20)
            plt.scatter(final_pca[:, 0], final_pca[:, 1], alpha=0.6, 
                       label='Final', color='blue', s=20)
            
            for i in range(min(20, len(initial))):
                plt.arrow(initial_pca[i, 0], initial_pca[i, 1],
                         final_pca[i, 0] - initial_pca[i, 0],
                         final_pca[i, 1] - initial_pca[i, 1],
                         head_width=0.02, head_length=0.02, fc='gray', ec='gray', alpha=0.5)
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title(f'{key}: PCA Visualization')
            plt.legend()
        
        plt.subplot(2, 3, 5)
        initial_norms = np.linalg.norm(initial, axis=1)
        final_norms = np.linalg.norm(final, axis=1)
        
        plt.scatter(initial_norms, final_norms, alpha=0.6)
        plt.plot([initial_norms.min(), initial_norms.max()], 
                [initial_norms.min(), initial_norms.max()], 'r--', alpha=0.8)
        plt.xlabel('Initial Norm')
        plt.ylabel('Final Norm')
        plt.title(f'{key}: Vector Norm Changes')
        
        plt.subplot(2, 3, 6)
        plt.axis('off')
        stats_text = f"""
        Codebook: {key}
        Shape: {diff_data['initial_shape']}
        
        L2 Distance:
        Mean: {diff_data['mean_l2_distance']:.4f}
        Std: {diff_data['std_l2_distance']:.4f}
        Max: {diff_data['max_l2_distance']:.4f}
        Min: {diff_data['min_l2_distance']:.4f}
        
        Cosine Similarity:
        Mean: {diff_data['mean_cosine_similarity']:.4f}
        Std: {diff_data['std_cosine_similarity']:.4f}
        """
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'codebook_analysis_{key}.pdf')
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        except Exception as e:
            print(f"✗ Failed to save visualization: {str(e)}")
        
        plt.close()


def analyze_codebook_usage(model, test_loader, device):
    model.eval()
    codebook_usage = {}
    
    with torch.no_grad():
        for batch_data in test_loader:
            x, adj, batch_num_nodes = batch_data
            x = x.to(device)
            adj = adj.to(device)
            batch_num_nodes = batch_num_nodes.tolist()
            
            results = model(x, adj, batch_num_nodes, return_intermediate=True)
            indices = results['quantized_indices']
            
            unique_indices, counts = torch.unique(indices, return_counts=True)

            for idx, count in zip(unique_indices.cpu().numpy(), counts.cpu().numpy()):
                if idx not in codebook_usage:
                    codebook_usage[idx] = 0
                codebook_usage[idx] += count
    
    return codebook_usage


def save_codebook_analysis_report(differences, codebook_usage, output_dir):
    report_path = os.path.join(output_dir, 'codebook_analysis_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Codebook Change Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. Codebook Difference Statistics:\n")
        f.write("-" * 40 + "\n")
        for key, diff_data in differences.items():
            f.write(f"\nCodebook: {key}\n")
            f.write(f"  Shape: {diff_data['initial_shape']}\n")
            f.write(f"  Mean L2 distance: {diff_data['mean_l2_distance']:.6f}\n")
            f.write(f"  L2 distance std: {diff_data['std_l2_distance']:.6f}\n")
            f.write(f"  Max L2 distance: {diff_data['max_l2_distance']:.6f}\n")
            f.write(f"  Min L2 distance: {diff_data['min_l2_distance']:.6f}\n")
            f.write(f"  Mean cosine similarity: {diff_data['mean_cosine_similarity']:.6f}\n")
            f.write(f"  Cosine similarity std: {diff_data['std_cosine_similarity']:.6f}\n")
        
        f.write(f"\n\n2. Codebook Usage Statistics:\n")
        f.write("-" * 40 + "\n")
        if codebook_usage:
            total_usage = sum(codebook_usage.values())
            f.write(f"Total usage count: {total_usage}\n")
            f.write(f"Number of used codewords: {len(codebook_usage)}\n")
            
            sorted_usage = sorted(codebook_usage.items(), key=lambda x: x[1], reverse=True)
            f.write(f"\nTop 10 most frequently used codewords:\n")
            for i, (idx, count) in enumerate(sorted_usage[:10]):
                f.write(f"  {i+1}. Codeword {idx}: {count} times ({count/total_usage*100:.2f}%)\n")
    
    print(f"Analysis report saved: {report_path}")


def main():
    print("=== Graph Motif Quantizer Joint Training ===")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description='Train Graph Motif Quantizer')
    parser.add_argument('--dataset', type=str, default='BZR', 
                       help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=25, 
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, 
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device (auto/cpu/cuda, default: auto)')
    parser.add_argument('--output_dir', type=str, default='./outputs', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"motif_quantizer_{timestamp}")
    log_dir = os.path.join(output_dir, "logs")
    save_dir = os.path.join(output_dir, "checkpoints")
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Log directory: {log_dir}")
    print(f"Checkpoint directory: {save_dir}")
    
    try:
        print("1. Loading data...")
        train_loader, val_loader, input_dim, max_nodes = create_data_loaders(
            dataset_name=args.dataset, 
            batch_size=args.batch_size,
            train_ratio=0.8
        )
        
        print("2. Configuring model...")
        config = {
            'hidden_dim': 256,
            'embedding_dim': 4096,
            'num_layers': 2,
            'assign_hidden_dim': 32,
            'assign_ratio': 0.5,
            'num_pooling': 2,
            
            'codebook_size': 512,
            'codebook_type': 'euclidean',
            'num_codebooks': 1,
            'decay': 0.9,
            'threshold_ema_dead_code': 5,
            'sample_codebook_temp': 0.0,
            'gaussian_delta': 1.0,
            
            'unpooling_method': 'mlp_transform',
            'commitment_weight': 1.0,
            'reconstruction_weight': 1,
            'structure_weight': 0.3,
            'vq_weight': 1.0,
            'dropout': 0.1,
            'linkpred_weight': 1,
            'linkpred': True,
            'bn': True
        }
        
        print("Model configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("3. Creating model...")
        model = create_model(input_dim, max_nodes, config)
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        print("3.1. Saving initial codebooks...")
        initial_codebooks = save_initial_codebooks(model, save_dir)
        print(f"Initial codebooks saved: {len(initial_codebooks)} codebooks")
        
        print("4. Starting training...")
        trained_model = train_graph_motif_quantizer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device,
            log_dir=log_dir,
            save_dir=save_dir,
            log_interval=5,
            val_interval=10,
        )
        
        print("5. Analyzing codebook changes...")
        final_codebooks = extract_final_codebooks(trained_model)
        differences = compute_codebook_differences(initial_codebooks, final_codebooks)
        
        codebook_usage = analyze_codebook_usage(trained_model, val_loader, device)
        
        vis_dir = os.path.join(output_dir, "codebook_analysis")
        visualize_codebook_changes(initial_codebooks, final_codebooks, differences, vis_dir)
        
        save_codebook_analysis_report(differences, codebook_usage, vis_dir)
    
        print("6. Testing model...")
        test_losses, sample_results = test_model(trained_model, val_loader, device)
        
        print("7. Saving results...")
        final_save_path = os.path.join(save_dir, 'final_model.pth')
        torch.save({
            'model_state_dict': trained_model.state_dict(),
            'config': config,
            'test_losses': test_losses,
            'input_dim': input_dim,
            'max_nodes': max_nodes,
            'initial_codebooks': initial_codebooks,
            'final_codebooks': final_codebooks,
            'codebook_differences': differences,
            'codebook_usage': codebook_usage
        }, final_save_path)
        
        print("=" * 60)
        print("✓ Training completed!")
        print(f"\nOutput files:")
        print(f"  - Model checkpoints: {save_dir}")
        print(f"  - Training logs: {log_dir}")
        print(f"  - Final model: {final_save_path}")
        print(f"  - Codebook analysis: {vis_dir}")
        
        print(f"\n=== Codebook Change Summary ===")
        for key, diff_data in differences.items():
            print(f"{key}:")
            print(f"  Mean L2 distance: {diff_data['mean_l2_distance']:.4f}")
            print(f"  Mean cosine similarity: {diff_data['mean_cosine_similarity']:.4f}")
        
        if codebook_usage:
            print(f"\nCodebook usage statistics:")
            print(f"  Number of used codewords: {len(codebook_usage)}")
            print(f"  Total usage count: {sum(codebook_usage.values())}")
        
        visualize_training_progress(log_dir)
        
        print(f"\n=== Usage Example ===")
        print("Load trained model:")
        print(f"```python")
        print(f"checkpoint = torch.load('{final_save_path}')")
        print(f"model = create_model(checkpoint['input_dim'], checkpoint['max_nodes'], checkpoint['config'])")
        print(f"model.load_state_dict(checkpoint['model_state_dict'])")
        print(f"model.eval()")
        print(f"")
        print(f"# Get graph motif tokens")
        print(f"motif_results = model.get_motif_tokens(x, adj, batch_num_nodes)")
        print(f"tokens = motif_results['tokens']")
        print(f"```")
        
    except Exception as e:
        print(f"✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
