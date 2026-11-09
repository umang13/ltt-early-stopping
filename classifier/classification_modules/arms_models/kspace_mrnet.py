import pdb
import torch
import torch.nn as nn
from torchvision import models
import numpy as np

from classifier.utils import transforms
from classifier.utils.utils import string_to_list

from .kspace_net import kSpaceNet
from .fft_conv import FFTConv2d


class kspaceMRNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        print("*" * 20, "Initializing kspaceMRNet", "*" * 20)

        self.num_classes = config.num_classes
        self.num_labels = config.num_labels

        self.label_names = string_to_list(config.label_names)
        self.config = config

        backbone, num_ftrs = self.get_backbone()

        self.conv_kspace = FFTConv2d(1, config.fft_conv_out, kernel_size=7, stride=1, bias=False)
        self.layernorm = nn.LayerNorm(
            elementwise_affine=False, normalized_shape=num_ftrs
        )  
        self.backbone = backbone
        self.dropout = nn.Dropout2d(p=self.config.dropout)

        if self.config.pool_type == "avg_pool":
            self.gap = nn.AdaptiveAvgPool2d(1)
        elif self.config.pool_type == "max_pool":
            self.gap = nn.AdaptiveMaxPool2d(1)
        elif self.config.pool_type == "attention":
            # Attention pooling mechanism
            # Learnable query vector that will attend to all slice features
            self.attention_query = nn.Parameter(torch.randn(1, num_ftrs))  # [1, feature_dim]
            self.attention_key = nn.Linear(num_ftrs, num_ftrs)
            self.attention_value = nn.Linear(num_ftrs, num_ftrs)
            self.attention_scale = 1.0 / (num_ftrs ** 0.5)  # Scale factor for attention
            self.attention_dropout = nn.Dropout(p=self.config.dropout)
        elif self.config.pool_type == "learned_weight":
            # Simple learned importance weighting mechanism
            # Network that outputs a single scalar importance score per slice
            # Default: 2-layer MLP with ReLU activation
            hidden_dim = getattr(self.config, 'importance_hidden_dim', num_ftrs // 2)
            self.importance_network = nn.Sequential(
                nn.Linear(num_ftrs, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=self.config.dropout),
                nn.Linear(hidden_dim, 1)  # Output single scalar per slice
            )
        elif self.config.pool_type == "class_specific_importance":
            # Class-specific importance networks: one importance network per class label
            # Each network learns which slices are important for its specific classification task
            hidden_dim = getattr(self.config, 'importance_hidden_dim', num_ftrs // 2)
            self.class_importance_networks = nn.ModuleList()
            for _ in range(self.config.num_labels):
                importance_network = nn.Sequential(
                    nn.Linear(num_ftrs, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=self.config.dropout),
                    nn.Linear(hidden_dim, 1)  # Output single scalar per slice
                )
                self.class_importance_networks.append(importance_network)
        else:
            raise NotImplementedError(
                f"Pool type {self.config.pool_type} not implemented"
            )

        # List of linear layers for multiple output heads
        self.label_layers = (
            nn.ModuleList()
        )
        # self.slice_label_layers = (
        #     nn.ModuleList()
        # )
        # for _ in range(0, self.config.num_labels):
        #     self.slice_label_layers.append(nn.Linear(num_ftrs, self.config.num_classes))

        for _ in range(0, self.config.num_labels):
            self.label_layers.append(nn.Linear(num_ftrs, self.config.num_classes))

    def get_backbone(self):
        if self.config.pretrained == "imagenet":
            pretrained = True
        else:
            pretrained = False

        if self.config.slice_model_type == "alexnet":
            num_ftrs = 256

            weights = "IMAGENET1K_V1" if pretrained else None
            backbone = models.alexnet(weights=weights).features
            # conv_kspace = FFTConv2d(1, 1, kernel_size=5, stride=1, bias=False)
            # layernorm = nn.LayerNorm(
            #     elementwise_affine=False, normalized_shape=self.config.image_shape
            # )

        elif self.config.slice_model_type == "resnet50":
            num_ftrs = 2048

            weights = "IMAGENET1K_V1" if pretrained else None
            resnet50 = models.resnet50(weights=weights)

            backbone = nn.Sequential(*list(resnet50.children())[:-1])
            # conv_kspace = FFTConv2d(1, 1, kernel_size=5, stride=1, bias=False)
            # layernorm = nn.LayerNorm(
            #     elementwise_affine=False, normalized_shape=self.config.image_shape
            # )
        elif self.config.slice_model_type == "resnet18":
            weights = "IMAGENET1K_V1" #if pretrained else None
            resnet18 = models.resnet18(weights=weights)
            # resnet18.maxpool = nn.Identity()
            # resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            # # # resnet18.maxpool = nn.Identity()
            # resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            backbone = nn.Sequential(*list(resnet18.children())[:-1])
            # conv_kspace = FFTConv2d(1, 1, kernel_size=5, stride=1, bias=False)
            # layernorm = nn.LayerNorm(elementwise_affine=False, normalized_shape=self.config.image_shape)
            num_ftrs = 512
        else:
            raise NotImplementedError(
                f"Slice model {self.config.slice_model_type} not implemented"
            )

        return backbone, num_ftrs

    def get_center_bias(self, n_slices, device, bias_strength=None):
        """
        Generate positional bias with flat center region and lower weights at extremes.
        
        Center slices get the same high weight, only edge slices get lower weights.
        
        Args:
            n_slices: Number of slices
            device: Device to create tensor on
            bias_strength: Strength of center bias (default: 2.0, higher = stronger bias)
        
        Returns:
            bias: Tensor of shape [n_slices] with higher values for center slices
        """
        if bias_strength is None:
            bias_strength = getattr(self.config, 'center_bias_strength', 2.0)
        
        # Get center region fraction (default: 0.5 means middle 50% of slices)
        center_region_fraction = getattr(self.config, 'center_region_fraction', 0.5)
        
        # Create indices from 0 to n_slices-1
        indices = torch.arange(n_slices, dtype=torch.float32, device=device)
        
        # Define center region boundaries
        center = (n_slices - 1) / 2.0
        center_region_half_width = (n_slices * center_region_fraction) / 2.0
        center_start = center - center_region_half_width
        center_end = center + center_region_half_width
        
        # Initialize bias: center region gets high value, edges get lower
        bias = torch.ones(n_slices, dtype=torch.float32, device=device)
        
        # Find slices outside the center region (edge slices)
        left_edge_mask = indices < center_start
        right_edge_mask = indices > center_end
        
        # Compute distance from center region for edge slices
        # For left edge: distance from center_start
        # For right edge: distance from center_end
        left_distances = (center_start - indices[left_edge_mask]) / max(center_region_half_width, 1.0)
        right_distances = (indices[right_edge_mask] - center_end) / max(center_region_half_width, 1.0)
        
        # Apply decay function to edge slices (exponential decay)
        # Higher bias_strength = faster decay = more emphasis on center
        if left_distances.numel() > 0:
            bias[left_edge_mask] = torch.exp(-bias_strength * left_distances)
        if right_distances.numel() > 0:
            bias[right_edge_mask] = torch.exp(-bias_strength * right_distances)
        
        # Normalize so that the bias doesn't dominate (optional scaling)
        bias = bias * getattr(self.config, 'center_bias_scale', 1.0)
        
        return bias

    def feature_extract(self, kspace):
        debug = getattr(self.config, 'debug_forward', False)
        
        if debug:
            print(f"\n{'='*80}")
            print(f"DEBUG: feature_extract - Input kspace shape: {kspace.shape}")
            print(f"DEBUG: kspace stats - min: {kspace.abs().min():.6f}, max: {kspace.abs().max():.6f}, mean: {kspace.abs().mean():.6f}")
        
        # Single-coil only: handle volume and slice inputs
        if "vol" in self.config.dataset:
            # Expected shape: (B=1, nslice, H, W) or (B, nslice, H, W)
            kspace = kspace.squeeze(0)
            kspace = kspace.unsqueeze(1)
            if debug:
                print(f"DEBUG: After squeeze/unsqueeze - shape: {kspace.shape}, n_slices: {kspace.shape[0]}")
        else:
            # Expected shape: (B, H, W)
            batch_size, H, W = kspace.shape
            kspace = kspace.reshape(-1, 1, H, W)
            if debug:
                print(f"DEBUG: After reshape (sc) - shape: {kspace.shape}")

        # print("1. kspace.shape", kspace.shape)
        image_data = torch.fft.fftshift(
            torch.fft.ifftn(torch.fft.ifftshift(kspace, dim=(-2, -1)), dim=(-2, -1)),
            dim=(-2, -1),
        )
        kspace = torch.fft.fftn(image_data, dim=(-2, -1))
        # print("2. kspace.shape", kspace.shape)
        if debug:
            print(f"DEBUG: After FFT ops - kspace shape: {kspace.shape}")
            print(f"DEBUG: After FFT ops - kspace stats - min: {kspace.abs().min():.6f}, max: {kspace.abs().max():.6f}, mean: {kspace.abs().mean():.6f}")
            print(f"DEBUG: conv_kspace - kernel_size: {self.conv_kspace.kernel_size}, in_channels: {self.conv_kspace.in_channels}, out_channels: {self.conv_kspace.out_channels}")
            # Check conv_kspace weights
            weight = self.conv_kspace.weight
            print(f"DEBUG: conv_kspace weight stats - real min: {weight.real.min():.6f}, real max: {weight.real.max():.6f}, real mean: {weight.real.mean():.6f}")
            print(f"DEBUG: conv_kspace weight stats - imag min: {weight.imag.min():.6f}, imag max: {weight.imag.max():.6f}, imag mean: {weight.imag.mean():.6f}")
            print(f"DEBUG: conv_kspace weight abs - min: {weight.abs().min():.6f}, max: {weight.abs().max():.6f}, mean: {weight.abs().mean():.6f}")
            # Check if weights are all zeros or very small
            if weight.abs().max() < 1e-6:
                print(f"WARNING: conv_kspace weights appear to be zeros or extremely small!")
            if (weight.abs() < 1e-10).all():
                print(f"WARNING: conv_kspace weights are all near zero!")
        # print("kspace.shape in forward of kspaceMRNet", kspace.shape)
        out_complex = self.conv_kspace(kspace)
        # print(out_complex.shape)
        if debug:
            print(f"DEBUG: After conv_kspace - out_complex shape: {out_complex.shape}")
            print(f"DEBUG: out_complex stats - min: {out_complex.abs().min():.6f}, max: {out_complex.abs().max():.6f}, mean: {out_complex.abs().mean():.6f}")
        
        out_mag = out_complex.abs()
        # del out_complex, kspace, image_data;
        # print("out_mag.shape", out_mag.shape, self.config.image_shape)
        out_mag = transforms.center_crop(out_mag, self.config.image_shape)
        # out_mag = self.layernorm(out_mag)
        if debug:
            print(f"DEBUG: After center_crop - out_mag shape: {out_mag.shape}")
            print(f"DEBUG: out_mag stats - min: {out_mag.min():.6f}, max: {out_mag.max():.6f}, mean: {out_mag.mean():.6f}")
        
        if self.config.fft_conv_out == 1:
            out_mag = out_mag.repeat(1, 3, 1, 1)
            if debug:
                print(f"DEBUG: After repeat(1,3,1,1) - out_mag shape: {out_mag.shape}")
        
        out = self.backbone(out_mag)
        
        if debug:
            print(f"DEBUG: After backbone - out shape: {out.shape}")
            print(f"DEBUG: out stats - min: {out.min():.6f}, max: {out.max():.6f}, mean: {out.mean():.6f}, std: {out.std():.6f}")
            print(f"DEBUG: out has NaN: {torch.isnan(out).any()}, has Inf: {torch.isinf(out).any()}")
            print(f"{'='*80}\n")
        
        return out

    def compute_slice_predictions(self, slice_feats):
        """
        Compute slice-level predictions and logits from slice features.
        
        Args:
            slice_feats: Tensor of shape [n_slices, feature_dim] containing features for each slice
        
        Returns:
            slice_logits_volume: Dict mapping label names to tensors of shape [n_slices, num_classes]
                                computed using volume model's label_layers
            slice_logits_slice: Dict mapping label names to tensors of shape [n_slices, num_classes]
                               computed using slice classifier's label_layers
        """
        # Compute slice-level logits using volume model's label_layers
        # slice_feats shape: [n_slices, feature_dim]
        slice_logits_volume = {}
        for i in range(0, len(self.label_layers)):
            # Process all slices at once: [n_slices, feature_dim] -> [n_slices, num_classes]
            slice_logits_volume[self.label_names[i]] = self.label_layers[i](slice_feats)
        
        # Compute slice-level logits using slice classifier's label_layers
        slice_logits_slice = {}
        if hasattr(self, 'slice_label_layers'):
            # Use slice-level label layers if available
            slice_label_names = getattr(self, 'slice_label_names', self.label_names)
            for i in range(0, len(self.slice_label_layers)):
                label_name = slice_label_names[i] if i < len(slice_label_names) else self.label_names[i]
                # Process all slices at once for efficiency
                # slice_feats shape: [n_slices, feature_dim]
                slice_logits_slice[label_name] = self.slice_label_layers[i](slice_feats)  # [n_slices, num_classes]
        else:
            # Fallback: use volume model's label_layers if slice_label_layers not available
            print("WARNING: slice_label_layers not found, using volume model's label_layers for slice predictions")
            for i in range(0, len(self.label_layers)):
                slice_logits_slice[self.label_names[i]] = self.label_layers[i](slice_feats)
        
        return slice_logits_volume, slice_logits_slice

    def forward(self, kspace, ret_slice_feats=False):
        debug = getattr(self.config, 'debug_forward', False)
        
        if debug:
            print(f"\n{'#'*80}")
            print(f"DEBUG: forward - Input kspace shape: {kspace.shape}")
            print(f"{'#'*80}")
        
        out = self.feature_extract(kspace)
        
        if debug:
            print(f"DEBUG: forward - After feature_extract, out shape: {out.shape}")
            print(f"DEBUG: forward - Dropout type: {type(self.dropout)}, p: {self.dropout.p}")
        # print("out.shape at predropout kspaceMRNet", out.shape)
        # out = out.float()
        # out = self.dropout(out)
        
        if debug:
            print(f"DEBUG: forward - After dropout, out shape: {out.shape}")
            print(f"DEBUG: forward - out stats after dropout - min: {out.min():.6f}, max: {out.max():.6f}, mean: {out.mean():.6f}")
        
        out = out.view(out.size(0), -1)
        slice_feats = out.detach().clone()
        # out = out.detach().requires_grad_(False)
        out = self.layernorm(out)
        if debug:
            print(f"DEBUG: forward - After view, out shape: {out.shape}")
            print(f"DEBUG: forward - n_slices: {out.shape[0]}, feature_dim: {out.shape[1]}")
            print(f"DEBUG: forward - Pool type: {self.config.pool_type}")
            print(f"DEBUG: variance across slices: {out.var(dim=0).mean():.6f}")
        
        # Initialize variable for class-specific aggregated features (only used for class_specific_importance)
        class_specific_aggregated_features = None
        
        if self.config.pool_type == "avg_pool":
            out = torch.mean(out, 0, keepdim=True)
            if debug:
                print(f"DEBUG: forward - After avg_pool (dim=0), out shape: {out.shape}")
        elif self.config.pool_type == "max_pool":
            out = torch.max(out, 0, keepdim=True)[0]
            if debug:
                print(f"DEBUG: forward - After max_pool (dim=0), out shape: {out.shape}")
        elif self.config.pool_type == "attention":
            # Attention pooling: out shape is [n_slices, feature_dim]
            n_slices, feature_dim = out.shape
            
            # Compute key and value projections from slice features
            key = self.attention_key(out)  # [n_slices, feature_dim]
            value = self.attention_value(out)  # [n_slices, feature_dim]
            
            # Use learnable query vector: [1, feature_dim]
            query = self.attention_query  # [1, feature_dim]
            
            # Compute attention scores: query @ key^T / sqrt(d_k)
            # query: [1, feature_dim], key: [n_slices, feature_dim]
            # attention_scores: [1, n_slices]
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.attention_scale
            
            # Add center bias to favor center slices
            center_bias = self.get_center_bias(n_slices, out.device)
            attention_scores = attention_scores + center_bias.unsqueeze(0)  # [1, n_slices]
            
            # Apply softmax to get attention weights (normalized across slices)
            attention_weights = torch.softmax(attention_scores, dim=-1)  # [1, n_slices]
            attention_weights = self.attention_dropout(attention_weights)
            
            # Apply attention to values: attention_weights @ value
            # attention_weights: [1, n_slices], value: [n_slices, feature_dim]
            # Result: [1, feature_dim] - weighted sum of slice features
            out = torch.matmul(attention_weights, value)  # [1, feature_dim]
            
            if debug:
                print(f"DEBUG: forward - After attention pooling, out shape: {out.shape}")
                print(f"DEBUG: forward - Attention weights stats - min: {attention_weights.min():.6f}, max: {attention_weights.max():.6f}, mean: {attention_weights.mean():.6f}")
                print(f"DEBUG: forward - Attention weights (sum should be 1.0): {attention_weights.sum().item():.6f}")
                print(f"DEBUG: forward - Top 3 slice attention weights: {torch.topk(attention_weights.squeeze(), k=min(3, n_slices)).values.tolist()}")
        elif self.config.pool_type == "learned_weight":
            # Learned importance weighting: out shape is [n_slices, feature_dim]
            n_slices, feature_dim = out.shape
            
            # Compute importance scores for each slice
            # importance_scores: [n_slices, 1]
            importance_scores = self.importance_network(out)  # [n_slices, 1]
            
            # Add center bias to favor center slices
            center_bias = self.get_center_bias(n_slices, out.device)
            importance_scores = importance_scores.squeeze(-1) + center_bias  # [n_slices]
            
            # Apply softmax to normalize importance scores to weights (sum to 1)
            importance_weights = torch.softmax(importance_scores, dim=0)  # [n_slices]
            
            # Compute weighted sum of slice features
            # importance_weights: [n_slices], out: [n_slices, feature_dim]
            # Result: [feature_dim] -> [1, feature_dim]
            out = torch.sum(importance_weights.unsqueeze(-1) * out, dim=0, keepdim=True)  # [1, feature_dim]
            
            if debug:
                print(f"DEBUG: forward - After learned weight pooling, out shape: {out.shape}")
                print(f"DEBUG: forward - Importance scores stats - min: {importance_scores.min():.6f}, max: {importance_scores.max():.6f}, mean: {importance_scores.mean():.6f}")
                print(f"DEBUG: forward - Importance weights stats - min: {importance_weights.min():.6f}, max: {importance_weights.max():.6f}, mean: {importance_weights.mean():.6f}")
                print(f"DEBUG: forward - Importance weights (sum should be 1.0): {importance_weights.sum().item():.6f}")
                print(f"DEBUG: forward - Top 3 slice importance weights: {torch.topk(importance_weights, k=min(3, n_slices)).values.tolist()}")
        elif self.config.pool_type == "class_specific_importance":
            # Class-specific importance pooling: each label gets its own aggregated features
            # Store original slice features for class-specific pooling
            class_specific_aggregated_features = []
            n_slices, feature_dim = out.shape
            
            # For each label, use its corresponding importance network to aggregate features
            for label_idx in range(len(self.class_importance_networks)):
                # Compute importance scores for this class using its specific network
                # importance_scores: [n_slices, 1]
                importance_scores = self.class_importance_networks[label_idx](out)  # [n_slices, 1]
                
                # Add center bias to favor center slices
                center_bias = self.get_center_bias(n_slices, out.device)
                importance_scores = importance_scores.squeeze(-1) + center_bias  # [n_slices]
                
                # Apply softmax to normalize importance scores to weights (sum to 1)
                importance_weights = torch.softmax(importance_scores, dim=0)  # [n_slices]
                
                # Compute weighted sum of slice features for this class
                # importance_weights: [n_slices], out: [n_slices, feature_dim]
                # Result: [feature_dim] -> [1, feature_dim]
                aggregated_feat = torch.sum(importance_weights.unsqueeze(-1) * out, dim=0, keepdim=True)  # [1, feature_dim]
                class_specific_aggregated_features.append(aggregated_feat)
                
                if debug:
                    print(f"DEBUG: forward - Class {label_idx} ({self.label_names[label_idx]}) - After class-specific importance pooling, aggregated_feat shape: {aggregated_feat.shape}")
                    print(f"DEBUG: forward - Class {label_idx} - Importance weights stats - min: {importance_weights.min():.6f}, max: {importance_weights.max():.6f}, mean: {importance_weights.mean():.6f}")
                    print(f"DEBUG: forward - Class {label_idx} - Importance weights (sum should be 1.0): {importance_weights.sum().item():.6f}")
            
            # Store class-specific aggregated features for use in label layer computation
            # We'll handle this in the label layer loop below
            # Note: For class_specific_importance, 'out' still contains slice features [n_slices, feature_dim]
            # The aggregated features are stored in class_specific_aggregated_features
        else:
            raise NotImplementedError(f"Pool type {self.config.pool_type} not implemented")

        if debug:
            if self.config.pool_type == "class_specific_importance":
                print(f"DEBUG: forward - After pooling, out shape (still slice features): {out.shape}")
                print(f"DEBUG: forward - Number of class-specific aggregated features: {len(class_specific_aggregated_features)}")
            else:
                print(f"DEBUG: forward - After pooling, out shape: {out.shape}")
                print(f"DEBUG: forward - out stats after pooling - min: {out.min():.6f}, max: {out.max():.6f}, mean: {out.mean():.6f}")

        out_dict = {}

        for i in range(0, len(self.label_layers)):
            # For class_specific_importance, use class-specific aggregated features
            # For other pool types, use the shared pooled features (out)
            if self.config.pool_type == "class_specific_importance":
                # Use the pre-computed class-specific aggregated features
                class_feat = class_specific_aggregated_features[i]  # [1, feature_dim]
                logits = self.label_layers[i](class_feat)
            else:
                # Use shared pooled features for all labels
                logits = self.label_layers[i](out)
            out_dict[self.label_names[i]] = logits
            if debug:
                print(f"DEBUG: forward - {self.label_names[i]} logits shape: {logits.shape}, values: {logits.squeeze().tolist()}")
        # if ret_slice_feats:
        # # Compute predictions using volume-level pooled features but with slice-level label layers
        #     volume_pooled_slice_label_logits = {}
        #     if hasattr(self, 'slice_label_layers') :
        #         slice_label_names = getattr(self, 'slice_label_names', self.label_names)
        #         for i in range(0, len(self.slice_label_layers)):
        #             label_name = slice_label_names[i] if i < len(slice_label_names) else self.label_names[i]
        #             logits = self.slice_label_layers[i](out)  # out is [1, feature_dim] after pooling
        #             volume_pooled_slice_label_logits[label_name] = logits
        #             if debug:
        #                 print(f"DEBUG: forward - {label_name} (volume pooled + slice label layers) logits shape: {logits.shape}, values: {logits.squeeze().tolist()}")
        #     else:
        #         if debug:
        #             print("DEBUG: forward - WARNING: slice_label_layers not found, skipping volume_pooled_slice_label_logits")

        if debug:
            print(f"{'#'*80}\n")
        if ret_slice_feats:
            slice_logits_volume, slice_logits_slice = self.compute_slice_predictions(slice_feats)
            return out_dict, slice_feats, slice_logits_volume, slice_logits_slice, None
        else:
            return out_dict
