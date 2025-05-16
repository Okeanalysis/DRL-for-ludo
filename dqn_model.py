"""
Deep Q-Network (DQN) Model Architecture

This module implements neural network architectures for DQN:
- Standard DQN with fully connected layers
- Dueling DQN architecture that separates state value and advantage estimation
- CNN-based architectures for visual inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQNModel(nn.Module):
    """Standard Deep Q-Network architecture with fully connected layers."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """Initialize the Q-Network.
        
        Args:
            state_dim: Dimensions of the state space (int or tuple)
            action_dim: Number of possible actions
            hidden_dim: Size of hidden layers
        """
        super(DQNModel, self).__init__()
        
        # Handle different state dimension formats
        if isinstance(state_dim, tuple):
            # Flatten the input dimensions for FC layers
            input_dim = np.prod(state_dim)
        else:
            input_dim = state_dim
        
        # Define network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values for each action
        """
        # Ensure input is properly shaped
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten if needed
        
        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQNModel(nn.Module):
    """Dueling DQN architecture that separates state value and advantage estimation."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """Initialize the Dueling Q-Network.
        
        Args:
            state_dim: Dimensions of the state space (int or tuple)
            action_dim: Number of possible actions
            hidden_dim: Size of hidden layers
        """
        super(DuelingDQNModel, self).__init__()
        
        # Handle different state dimension formats
        if isinstance(state_dim, tuple):
            # Flatten the input dimensions for FC layers
            input_dim = np.prod(state_dim)
        else:
            input_dim = state_dim
        
        # Feature extraction layers (shared)
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        """Forward pass through the dueling network architecture.
        
        Args:
            x: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values for each action combining value and advantage estimates
        """
        # Ensure input is properly shaped
        if x.dim() > 2:
            x = x.view(x.size(0), -1)  # Flatten if needed
        
        # Extract features
        features = self.feature_layer(x)
        
        # Calculate value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using the dueling architecture formula:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # This ensures identifiability by forcing the advantage estimates to have zero mean
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class CNNDQNModel(nn.Module):
    """CNN-based DQN for visual input states such as images or game screens."""
    
    def __init__(self, input_shape, action_dim):
        """Initialize the CNN Q-Network.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            action_dim: Number of possible actions
        """
        super(CNNDQNModel, self).__init__()
        
        self.input_shape = input_shape
        c, h, w = input_shape
        
        # Feature extraction with CNN
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the flattened features
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        linear_input_size = conv_h * conv_w * 64
        
        # Q-value prediction with fully connected layers
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, action_dim)
        
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Q-values for each action
        """
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the CNN output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class DuelingCNNDQNModel(nn.Module):
    """Dueling CNN-based DQN for visual input states."""
    
    def __init__(self, input_shape, action_dim):
        """Initialize the Dueling CNN Q-Network.
        
        Args:
            input_shape: Shape of input images (channels, height, width)
            action_dim: Number of possible actions
        """
        super(DuelingCNNDQNModel, self).__init__()
        
        self.input_shape = input_shape
        c, h, w = input_shape
        
        # Feature extraction with CNN
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of the flattened features
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conv_h = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, 8, 4), 4, 2), 3, 1)
        conv_w = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, 8, 4), 4, 2), 3, 1)
        linear_input_size = conv_h * conv_w * 64
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        
    def forward(self, x):
        """Forward pass through the dueling CNN network.
        
        Args:
            x: Input state tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Q-values for each action combining value and advantage estimates
        """
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten the CNN output
        x = x.view(x.size(0), -1)
        
        # Calculate value and advantages
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine value and advantages
        return value + (advantages - advantages.mean(dim=1, keepdim=True))