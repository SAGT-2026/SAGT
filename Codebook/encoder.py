import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np


class StraightThroughThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=0.5):
        return (input > threshold).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
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

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

        if bn:
            self.bn_first = nn.BatchNorm1d(hidden_dim)
            self.bn_block = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 2)])
            self.bn_last = nn.BatchNorm1d(embedding_dim)

        self.pred_model = self.build_pred_layers(
            self.pred_input_dim, pred_hidden_dims, num_classes=1, num_aggs=self.num_aggs
        )

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last
    
    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model
    
    def construct_mask(self, max_nodes, batch_num_nodes, device=None): 
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).to(device)

    def apply_bn(self, x, bn_module):
        batch_size, num_nodes, feat_dim = x.shape
        x_reshaped = x.view(-1, feat_dim)
        x_bn = bn_module(x_reshaped)
        return x_bn.view(batch_size, num_nodes, feat_dim)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None,
                    bn_first=None, bn_block=None, bn_last=None):
        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn and bn_first is not None:
            x = self.apply_bn(x, bn_first)
        x_all = [x]

        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn and bn_block is not None and i < len(bn_block):
                x = self.apply_bn(x, bn_block[i])
            x_all.append(x)
        x = conv_last(x, adj)
        if self.bn and bn_last is not None:
            x = self.apply_bn(x, bn_last)
        x_all.append(x)

        if self.concat:
            x_tensor = torch.cat(x_all, dim=2)
        else:
            x_tensor = x
            
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes, device=x.device)
        else:
            self.embedding_mask = None

        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x, self.bn_first)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x, self.bn_block[i])
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj)
        if self.bn:
            x = self.apply_bn(x, self.bn_last)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, num_layers,
            assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
            pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True,
            assign_input_dim=-1, args=None):

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim,
                num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        self.final_cluster_features = None
        self.final_cluster_adj = None
        self.pooling_assignments = []

        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        self.bn_first_after_pool = nn.ModuleList()
        self.bn_block_after_pool = nn.ModuleList()
        self.bn_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                    self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                    add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)
            if bn:
                self.bn_first_after_pool.append(nn.BatchNorm1d(hidden_dim))
                self.bn_block_after_pool.append(nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 2)]))
                self.bn_last_after_pool.append(nn.BatchNorm1d(embedding_dim))

        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        
        assign_dim = 15
        
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                    assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                    normalize=True)
            if concat:
                assign_pred_input_dim = assign_hidden_dim * (assign_num_layers - 1) + assign_dim
            else:
                assign_pred_input_dim = assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)
    
    def gumbel_sigmoid_assignment(self, logits, temperature=1.0, threshold=0.5):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        y_soft = torch.sigmoid((logits + gumbel_noise) / temperature)
        y_hard = (y_soft > threshold).float()
        y = y_hard - y_soft.detach() + y_soft
        return y
    
    def normalize_assignment(self, assignment):
        eps = 1e-8
        row_sums = assignment.sum(dim=-1, keepdim=True) + eps
        return assignment / row_sums
    
    def forward(self, x, adj, batch_num_nodes, return_cluster_info=True, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes, device=x.device)
        else:
            embedding_mask = None

        embedding_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask,
                bn_first=self.bn_first, bn_block=self.bn_block, bn_last=self.bn_last)

        self.pooling_assignments = []
        
        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes, device=x.device)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(x_a, adj, 
                    self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i],
                    embedding_mask)

            assign_logits = self.assign_pred_modules[i](self.assign_tensor)
            self.assign_tensor = self.gumbel_sigmoid_assignment(
                    assign_logits, 1, 0.5
                )
            self.assign_tensor = self.normalize_assignment(self.assign_tensor)
                
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            self.pooling_assignments.append(self.assign_tensor.clone())

            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x
            
            embedding_tensor = self.gcn_forward(x, adj,
                    self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                    self.conv_last_after_pool[i],
                    bn_first=self.bn_first_after_pool[i] if self.bn else None,
                    bn_block=self.bn_block_after_pool[i] if self.bn else None,
                    bn_last=self.bn_last_after_pool[i] if self.bn else None)
            
            if i == self.num_pooling - 1:
                self.final_cluster_features = embedding_tensor
                self.final_cluster_adj = adj
            
        if return_cluster_info:
            return self.final_cluster_features, self.final_cluster_adj, self.pooling_assignments
        else:
            out, _ = torch.max(embedding_tensor, dim=1)
            return out

    def get_cluster_representations(self):
        if (self.final_cluster_features is None or 
            self.final_cluster_adj is None or 
            len(self.pooling_assignments) == 0):
            raise ValueError("Must call forward method first")
        
        return self.final_cluster_features, self.final_cluster_adj, self.pooling_assignments

    def get_pooling_assignment(self, layer_idx):
        if len(self.pooling_assignments) == 0:
            raise ValueError("Must call forward method first")
        
        if layer_idx < 0 or layer_idx >= len(self.pooling_assignments):
            raise ValueError(f"layer_idx must be between 0 and {len(self.pooling_assignments)-1}")
        
        return self.pooling_assignments[layer_idx]

    def get_layer_node_assignment(self, layer_idx, batch_num_nodes=None):
        assignment_matrix = self.get_pooling_assignment(layer_idx)
        dominant_probs, dominant_assignment = torch.max(assignment_matrix, dim=-1)
        
        if batch_num_nodes is not None:
            effective_num_nodes = batch_num_nodes if layer_idx == 0 else None
            
            if effective_num_nodes is not None:
                batch_size, max_nodes = dominant_assignment.shape
                for i, num_nodes in enumerate(effective_num_nodes):
                    dominant_assignment[i, num_nodes:] = -1
        
        return dominant_assignment, assignment_matrix

    def get_multi_layer_assignment(self, layer_idx, batch_num_nodes=None, threshold=0.1):
        assignment_matrix = self.get_pooling_assignment(layer_idx)
        batch_size, max_nodes, num_clusters = assignment_matrix.shape
        
        multi_assignment = []
        effective_num_nodes = batch_num_nodes if layer_idx == 0 else [max_nodes] * batch_size
        
        for batch_idx in range(batch_size):
            batch_assignment = []
            num_nodes = effective_num_nodes[batch_idx] if effective_num_nodes else max_nodes
            
            for node_idx in range(num_nodes):
                node_clusters = []
                for cluster_idx in range(num_clusters):
                    prob = assignment_matrix[batch_idx, node_idx, cluster_idx].item()
                    if prob > threshold:
                        node_clusters.append((cluster_idx, prob))
                
                node_clusters.sort(key=lambda x: x[1], reverse=True)
                batch_assignment.append(node_clusters)
            
            multi_assignment.append(batch_assignment)
        
        return multi_assignment

    def get_node_cluster_probabilities(self, batch_num_nodes=None, threshold=0.1):
        if len(self.pooling_assignments) == 0:
            raise ValueError("Must call forward method first")
        
        return self.get_pooling_assignment(-1)
    
    def get_dominant_cluster_assignment(self, batch_num_nodes=None):
        return self.get_layer_node_assignment(-1, batch_num_nodes)
    
    def get_multi_cluster_assignment(self, batch_num_nodes=None, threshold=0.1):
        return self.get_multi_layer_assignment(-1, batch_num_nodes, threshold)

    def get_hard_node_assignment(self, batch_num_nodes=None):
        import warnings
        warnings.warn("get_hard_node_assignment deprecated, use get_dominant_cluster_assignment", 
                     DeprecationWarning, stacklevel=2)
        dominant_assignment, _ = self.get_dominant_cluster_assignment(batch_num_nodes)
        return dominant_assignment