import torch
import torch.nn as nn

class MLP(nn.Module):
    """Simple MLP classifier - compatible with LocationEncoder framework"""
    def __init__(self, num_inputs, num_classes=1, dim_hidden=256, num_layers=3, dropout=False):
        """
        Args:
            num_inputs: Input dimension (embedding size from positional encoder)
            num_classes: Output dimension (1 for binary classification, >1 for multi-class)
            dim_hidden: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout: If True or float, adds dropout layers
        """
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(num_inputs, dim_hidden))
        layers.append(nn.ReLU())
        
        if dropout:
            dropout_p = dropout if isinstance(dropout, float) else 0.1
            layers.append(nn.Dropout(dropout_p))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(dim_hidden, dim_hidden))
            layers.append(nn.ReLU())
            
            if dropout:
                dropout_p = dropout if isinstance(dropout, float) else 0.1
                layers.append(nn.Dropout(dropout_p))
        
        # Output layer
        layers.append(nn.Linear(dim_hidden, num_classes))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, num_inputs)
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.net(x)
