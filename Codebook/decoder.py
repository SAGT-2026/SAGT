import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


class GraphDeconv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphDeconv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.bias = None
    
        self.reset_parameters()
    
    def reset_parameters(self):
        init.xavier_uniform_(self.weight.data, gain=nn.init.calculate_gain('relu'))
        if self.bias is not None:
            init.constant_(self.bias.data, 0.0)
    
    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        
        y = torch.matmul(x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        
        y = torch.matmul(adj, y)
        if self.add_self:
            y += torch.matmul(x, self.weight)
            if self.bias is not None:
                y += self.bias
        
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        
        return y


class SymmetricDecoder(nn.Module):
    def __init__(self, cluster_dim, hidden_dim, output_dim, num_pooling_layers=2,
                 gcn_layers_per_unpool=2, dropout=0.0, bn=True):
        super(SymmetricDecoder, self).__init__()
        self.cluster_dim = cluster_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_pooling_layers = num_pooling_layers
        self.gcn_layers_per_unpool = gcn_layers_per_unpool
        self.dropout = dropout
        self.bn = bn
        
        self.unpool_gcn_blocks = nn.ModuleList()
        
        current_dim = cluster_dim
        for pool_level in range(num_pooling_layers):
            gcn_block = nn.ModuleList()
            
            for gcn_idx in range(gcn_layers_per_unpool):
                if pool_level == 0 and gcn_idx == 0:
                    next_dim = hidden_dim
                elif pool_level == num_pooling_layers - 1 and gcn_idx == gcn_layers_per_unpool - 1:
                    next_dim = output_dim
                else:
                    next_dim = hidden_dim
                
                gcn_layer = GraphDeconv(
                    input_dim=current_dim,
                    output_dim=next_dim,
                    add_self=False,
                    normalize_embedding=(gcn_idx < gcn_layers_per_unpool - 1),
                    dropout=dropout
                )
                gcn_block.append(gcn_layer)
                current_dim = next_dim
            
            self.unpool_gcn_blocks.append(gcn_block)
            
            if pool_level < num_pooling_layers - 1:
                current_dim = hidden_dim
        
        if self.bn:
            self.bn_layers = nn.ModuleList()
            for pool_level in range(num_pooling_layers):
                bn_block = nn.ModuleList()
                for gcn_idx in range(gcn_layers_per_unpool):
                    if not (pool_level == num_pooling_layers - 1 and gcn_idx == gcn_layers_per_unpool - 1):
                        bn_dim = hidden_dim if pool_level < num_pooling_layers - 1 or gcn_idx < gcn_layers_per_unpool - 1 else output_dim
                        bn_block.append(nn.BatchNorm1d(bn_dim))
                    else:
                        bn_block.append(None)
                self.bn_layers.append(bn_block)
    
    def apply_bn(self, x, bn_layer):
        if bn_layer is None:
            return x
        batch_size, num_nodes, feat_dim = x.shape
        x_reshaped = x.view(-1, feat_dim)
        x_bn = bn_layer(x_reshaped)
        return x_bn.view(batch_size, num_nodes, feat_dim)
    
    def forward(self, cluster_features, cluster_adj, assignment_matrices, 
                target_adjs=None, original_node_counts=None):
        x = cluster_features
        adj = cluster_adj
        
        assignment_matrices_reversed = list(reversed(assignment_matrices))
        
        for pool_level in range(self.num_pooling_layers):
            if pool_level < len(assignment_matrices_reversed):
                S = assignment_matrices_reversed[pool_level]
                x = torch.bmm(S, x)
                adj = torch.bmm(torch.bmm(S, adj), S.transpose(1, 2))
            
            gcn_block = self.unpool_gcn_blocks[pool_level]
            bn_block = self.bn_layers[pool_level] if self.bn else None
            
            for gcn_idx, gcn_layer in enumerate(gcn_block):
                x = gcn_layer(x, adj)
                
                is_last_layer = (pool_level == self.num_pooling_layers - 1 and 
                               gcn_idx == self.gcn_layers_per_unpool - 1)
                if not is_last_layer:
                    x = F.relu(x)
                
                if self.bn and bn_block is not None:
                    x = self.apply_bn(x, bn_block[gcn_idx])
        
        return x, adj
    
    def get_reconstruction_loss(self, reconstructed_features, original_features, 
                               reconstructed_adj, original_adj, 
                               feature_weight=1.0, structure_weight=1.0):
        feature_loss = F.mse_loss(reconstructed_features, original_features)
        structure_loss = F.binary_cross_entropy_with_logits(
            reconstructed_adj, original_adj.float()
        )
        
        total_loss = feature_weight * feature_loss + structure_weight * structure_loss
        return total_loss, feature_loss, structure_loss


class GraphDecoder(nn.Module):
    def __init__(self, cluster_dim, hidden_dim, output_dim, num_layers, 
                 num_unpooling_layers=2,
                 dropout=0.0, bn=True):
        super(GraphDecoder, self).__init__()
        self.cluster_dim = cluster_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_unpooling_layers = num_unpooling_layers
        self.dropout = dropout
        self.bn = bn
        
        gcn_layers_per_unpool = max(1, num_layers // num_unpooling_layers)
            
        self.decoder = SymmetricDecoder(
            cluster_dim=cluster_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_pooling_layers=num_unpooling_layers,
            gcn_layers_per_unpool=gcn_layers_per_unpool,
            dropout=dropout,
            bn=bn)
    
    def forward(self, cluster_features, cluster_adj, assignment_matrices, 
                target_adjs=None, original_node_counts=None):
        return self.decoder(cluster_features, cluster_adj, assignment_matrices,
                          target_adjs, original_node_counts)
        
    def get_reconstruction_loss(self, reconstructed_features, originalFeatures, 
                               reconstructed_adj, original_adj, 
                               feature_weight=1.0, structure_weight=1.0):
        return self.decoder.get_reconstruction_loss(
            reconstructed_features, originalFeatures,
            reconstructed_adj, original_adj,
            feature_weight, structure_weight
        )


class GraphMotifQuantizer(nn.Module):
    def __init__(self, 
                 max_num_nodes, input_dim, hidden_dim, embedding_dim, 
                 num_layers, assign_hidden_dim, assign_ratio=0.25, num_pooling=1,
                 codebook_size=64, codebook_type='euclidean', num_codebooks=1,
                 decay=0.8, threshold_ema_dead_code=2, sample_codebook_temp=0.0,
                 gaussian_delta=1.0,
                 decoder_hidden_dim=None, decoder_num_layers=None,
                 unpooling_method='mlp_transform',
                 commitment_weight=0.25, reconstruction_weight=1.0, 
                 structure_weight=0.5, vq_weight=1.0, linkpred_weight=1.0,
                 dropout=0.0, bn=True, use_ddp=False, linkpred=True):
        super(GraphMotifQuantizer, self).__init__()
        
        from codebook_class import EuclideanCodebook, CosineSimCodebook
        from encoder import SoftPoolingGcnEncoder
        
        self.commitment_weight = commitment_weight
        self.reconstruction_weight = reconstruction_weight
        self.structure_weight = structure_weight
        self.vq_weight = vq_weight
        self.linkpred_weight = linkpred_weight
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        
        self.encoder = SoftPoolingGcnEncoder(
            max_num_nodes=max_num_nodes,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            assign_hidden_dim=assign_hidden_dim,
            assign_ratio=assign_ratio,
            num_pooling=num_pooling,
            concat=False,
            bn=bn,
            dropout=dropout,
            linkpred=linkpred
        )
        
        if codebook_type == 'euclidean':
            self.codebook = EuclideanCodebook(
                dim=embedding_dim,
                codebook_size=codebook_size,
                num_codebooks=num_codebooks,
                decay=decay,
                eps=1e-5,
                threshold_ema_dead_code=threshold_ema_dead_code,
                use_ddp=use_ddp,
                learnable_codebook=True,
                sample_codebook_temp=sample_codebook_temp,
                gaussian_delta=gaussian_delta
            )
        else:
            self.codebook = CosineSimCodebook(
                dim=embedding_dim,
                codebook_size=codebook_size,
                num_codebooks=num_codebooks,
                decay=decay,
                eps=1e-5,
                threshold_ema_dead_code=threshold_ema_dead_code,
                use_ddp=use_ddp,
                learnable_codebook=True,
                sample_codebook_temp=sample_codebook_temp,
                gaussian_delta=gaussian_delta
            )
        
        decoder_hidden_dim = decoder_hidden_dim or hidden_dim
        decoder_num_layers = decoder_num_layers or num_layers
        
        self.decoder = GraphDecoder(
            cluster_dim=embedding_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=input_dim,
            num_layers=decoder_num_layers,
            num_unpooling_layers=num_pooling,
            dropout=dropout,
            bn=bn
        )
        
        self.register_buffer('step', torch.tensor(0))
    
    def linkpred_loss(self, embedding, adj, batch_num_nodes, assignment_matrices=None, adj_hop=1):
        assign_tensor = assignment_matrices[0]
        batch_size = adj.shape[0]
        max_num_nodes = adj.shape[1]
        eps = 1e-7

        if batch_num_nodes is not None:
            mask = torch.zeros(batch_size, max_num_nodes, dtype=torch.bool, device=adj.device)
            for b, n in enumerate(batch_num_nodes):
                mask[b, :n] = True
        else:
            mask = torch.ones(batch_size, max_num_nodes, dtype=torch.bool, device=adj.device)

        pred_adj0 = torch.bmm(assign_tensor, assign_tensor.transpose(1, 2))

        pred_adj = pred_adj0
        tmp = pred_adj0
        for adj_pow in range(adj_hop - 1):
            tmp = torch.bmm(tmp, pred_adj0)
            pred_adj = pred_adj + tmp

        pred_adj = torch.min(pred_adj, torch.ones_like(pred_adj))

        adj_float = adj.float()
        link_loss = (-adj_float * torch.log(pred_adj + eps) -
                    (1 - adj_float) * torch.log(1 - pred_adj + eps))

        mask_2d = mask.unsqueeze(1) & mask.unsqueeze(2)
        link_loss = link_loss * mask_2d.float()
        num_entries = mask_2d.float().view(batch_size, -1).sum(dim=1).clamp(min=1)
        total_loss = link_loss.view(batch_size, -1).sum(dim=1) / num_entries

        return total_loss.mean()
    
    def create_cluster_mask(self, cluster_features, pooling_assignments, batch_num_nodes):
        batch_size, num_clusters = cluster_features.shape[:2]
        mask = torch.ones(batch_size, num_clusters, dtype=torch.bool, device=cluster_features.device)
        
        if pooling_assignments and len(pooling_assignments) > 0:
            final_assignment = pooling_assignments[-1]
            for batch_idx in range(batch_size):
                cluster_weights = final_assignment[batch_idx].sum(dim=0)
                threshold = 0.01
                mask[batch_idx] = cluster_weights > threshold
        
        return mask
    
    def quantize_cluster_features(self, cluster_features, mask=None):
        batch_size, num_clusters, feature_dim = cluster_features.shape
        
        flattened_features = cluster_features.view(1, -1, feature_dim)
        quantized_flat, indices_flat, distances_flat, embed = self.codebook(flattened_features)
        
        quantized_features = quantized_flat.view(batch_size, num_clusters, feature_dim)
        quantized_indices = indices_flat.view(batch_size, num_clusters)
        quantization_distances = distances_flat.view(batch_size, num_clusters, -1)
        
        if mask is not None:
            quantized_indices = quantized_indices.masked_fill(~mask, -1)
            quantized_features = quantized_features * mask.unsqueeze(-1).float()
        
        return quantized_features, quantized_indices, quantization_distances
    
    def forward(self, x, adj, batch_num_nodes=None, return_intermediate=False):
        encoder_results = self.encoder(x, adj, batch_num_nodes, return_cluster_info=True)
        
        if len(encoder_results) == 3:
            cluster_features, cluster_adj, assignment_matrices = encoder_results
            all_embeddings = None
        else:
            cluster_features, cluster_adj, assignment_matrices = encoder_results[:3]
            all_embeddings = encoder_results[3] if len(encoder_results) > 3 else None
        
        cluster_mask = self.create_cluster_mask(cluster_features, assignment_matrices, batch_num_nodes)
        
        quantized_features, quantized_indices, quantization_distances = self.quantize_cluster_features(
            cluster_features, mask=cluster_mask
        )
        
        reconstructed_x, reconstructed_adj = self.decoder(
            quantized_features, cluster_adj, assignment_matrices,
            target_adjs=[adj] * len(assignment_matrices),
            original_node_counts=batch_num_nodes
        )
        
        results = {
            'original_x': x,
            'original_adj': adj,
            'cluster_features': cluster_features,
            'quantized_features': quantized_features,
            'quantized_indices': quantized_indices,
            'cluster_adj': cluster_adj,
            'assignment_matrices': assignment_matrices,
            'cluster_mask': cluster_mask,
            'reconstructed_x': reconstructed_x,
            'reconstructed_adj': reconstructed_adj,
            'quantization_distances': quantization_distances,
            'all_embeddings': all_embeddings
        }
        
        if return_intermediate:
            return results
        else:
            return reconstructed_x, reconstructed_adj, quantized_indices
    
    def compute_losses(self, results):
        losses = {}
        
        recon_loss_total, recon_loss_feat, recon_loss_struct = self.decoder.get_reconstruction_loss(
            results['reconstructed_x'], results['original_x'],
            results['reconstructed_adj'], results['original_adj'],
            feature_weight=1.0, structure_weight=self.structure_weight
        )
        losses['reconstruction'] = recon_loss_total * self.reconstruction_weight
        losses['reconstruction_feature'] = recon_loss_feat
        losses['reconstruction_structure'] = recon_loss_struct
        
        cluster_features = results['cluster_features']
        quantized_features = results['quantized_features']
        
        commitment_loss = F.mse_loss(cluster_features, quantized_features.detach())
        losses['commitment'] = commitment_loss * self.commitment_weight

        vq_loss = F.mse_loss(cluster_features.detach(), quantized_features)
        losses['vq'] = vq_loss * self.vq_weight
        
        if self.linkpred:
            linkpred_loss = self.linkpred_loss(
                results['reconstructed_x'], 
                results['original_adj'], 
                None,
                assignment_matrices=results['assignment_matrices'],
                adj_hop=1
            )
            losses['linkpred'] = linkpred_loss * self.linkpred_weight
            
            if results['all_embeddings'] is not None and len(results['all_embeddings']) > 0:
                intermediate_linkpred_loss = self.linkpred_loss(
                    results['all_embeddings'][-1],
                    results['original_adj'],
                    None,
                    assignment_matrices=results['assignment_matrices'],
                    adj_hop=1
                )
                losses['linkpred_intermediate'] = intermediate_linkpred_loss * self.linkpred_weight * 0.5
        else:
            losses['linkpred'] = torch.tensor(0.0, device=cluster_features.device)
            losses['linkpred_intermediate'] = torch.tensor(0.0, device=cluster_features.device)

        total_loss = (losses['reconstruction'] + 
                        losses['commitment'] + 
                        losses['vq'] +
                        losses['linkpred'] + 
                        losses.get('linkpred_intermediate', 0))
        losses['total'] = total_loss
            
        return losses
    
    def training_step(self, batch, optimizer):
        x, adj, batch_num_nodes = batch[:3]
        
        results = self.forward(x, adj, batch_num_nodes, return_intermediate=True)
        losses = self.compute_losses(results)
        
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        
        self.step += 1
        return losses, results
    
    def validation_step(self, batch):
        with torch.no_grad():
            x, adj, batch_num_nodes = batch[:3]
            results = self.forward(x, adj, batch_num_nodes, return_intermediate=True)
            losses = self.compute_losses(results)
        return losses, results
    
    def get_motif_tokens(self, x, adj, batch_num_nodes=None):
        with torch.no_grad():
            results = self.forward(x, adj, batch_num_nodes, return_intermediate=True)
            return {
                'tokens': results['quantized_indices'],
                'mask': results['cluster_mask'],
                'cluster_features': results['cluster_features'],
                'quantized_features': results['quantized_features']
            }


def train_graph_motif_quantizer(model, train_dataloader, val_dataloader=None, 
                               num_epochs=100, lr=1e-3, device='cuda',
                               log_dir='./logs', save_dir='./checkpoints',
                               log_interval=10, val_interval=50):
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, f'run_{timestamp}'))
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    model = model.to(device)
    model.train()
    
    print(f"Starting training for {num_epochs} epochs")
    print(f"Log directory: {log_dir}")
    print(f"Model save directory: {save_dir}")
    print("=" * 60)
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_losses = {key: 0.0 for key in ['total', 'reconstruction', 'commitment', 'vq']}
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(train_dataloader):
            batch_x, batch_adj, batch_num_nodes = batch_data[:3]
            batch_x = batch_x.to(device)
            batch_adj = batch_adj.to(device)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)}")
            
            losses, results = model.training_step(
                (batch_x, batch_adj, batch_num_nodes), optimizer
            )
            
            for key, value in losses.items():
                if key in epoch_losses:
                    epoch_losses[key] += value.item()
            num_batches += 1
            global_step += 1
            
            if batch_idx % log_interval == 0:
                for key, value in losses.items():
                    writer.add_scalar(f'Train/{key}_loss', value.item(), global_step)
                
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, "
                      f"Loss: {losses['total'].item():.6f}, "
                      f"Recon: {losses['reconstruction'].item():.6f}, "
                      f"Commit: {losses['commitment'].item():.6f}, "
                      f"VQ: {losses['vq'].item():.6f}, "
                      f"LinkPred: {losses.get('linkpred', 0).item():.6f}")
        
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        if val_dataloader is not None and epoch % val_interval == 0:
            model.eval()
            val_losses = {key: 0.0 for key in ['total', 'reconstruction', 'commitment', 'vq']}
            val_batches = 0
            
            for val_batch in val_dataloader:
                val_x, val_adj, val_batch_num_nodes = val_batch[:3]
                val_x = val_x.to(device)
                val_adj = val_adj.to(device)
                
                losses, _ = model.validation_step((val_x, val_adj, val_batch_num_nodes))
                
                for key, value in losses.items():
                    if key in val_losses:
                        val_losses[key] += value.item()
                val_batches += 1
            
            for key in val_losses:
                val_losses[key] /= val_batches
                writer.add_scalar(f'Val/{key}_loss', val_losses[key], epoch)
            
            print(f"Validation - Loss: {val_losses['total']:.6f}, "
                  f"Recon: {val_losses['reconstruction']:.6f}")
            
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"Saved best model with validation loss: {best_val_loss:.6f}")
            
            model.train()
        
        scheduler.step()
        writer.add_scalar('Train/learning_rate', scheduler.get_last_lr()[0], epoch)
        
        for key, value in epoch_losses.items():
            writer.add_scalar(f'Epoch/{key}_loss', value, epoch)
        
        print(f"Epoch {epoch+1} completed - Average loss: {epoch_losses['total']:.6f}")
        
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'val_loss': best_val_loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print("Training completed!")
    return model
