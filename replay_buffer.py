"""
Replay Buffer Implementation for DRL Agents

This module provides replay memory implementations for DRL agents:
- Standard replay buffer for uniform sampling
- Prioritized experience replay (PER) buffer for non-uniform sampling based on TD errors
"""

import numpy as np
import random
from collections import deque, namedtuple
import operator


class ReplayBuffer:
    """Standard replay buffer with uniform sampling."""
    
    def __init__(self, capacity):
        """Initialize the replay buffer.
        
        Args:
            capacity: Maximum size of the replay buffer
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        """Sample a batch of experiences randomly.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of batched states, actions, rewards, next_states, and dones
        """
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)
        
        # Separate the batch into its components
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_full(self):
        """Check if buffer has reached capacity."""
        return len(self.buffer) == self.capacity


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer for efficient learning."""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        """Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum size of the replay buffer
            alpha: Exponent determining how much prioritization is used (0=no prioritization, 1=full prioritization)
            beta: Importance-sampling correction exponent
            beta_increment: Small value to increase beta over time towards 1
            epsilon: Small constant added to priorities to ensure non-zero probability
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # Small constant to ensure non-zero priority
        self.max_priority = 1.0  # Initial max priority for new experiences
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        # New experiences get max priority to ensure they're sampled at least once
        self.priorities[self.position] = self.max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample a batch of experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple containing:
            - Batched states, actions, rewards, next_states, and dones
            - Indices of sampled experiences
            - Importance sampling weights
        """
        # Calculate sampling probabilities based on priorities
        if len(self.buffer) < self.capacity:
            prob_distribution = self.priorities[:len(self.buffer)] / np.sum(self.priorities[:len(self.buffer)])
        else:
            prob_distribution = self.priorities / np.sum(self.priorities)
            
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=prob_distribution)
        
        # Get experiences for sampled indices
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        # Calculate importance sampling weights
        # Correct bias introduced by prioritized sampling using importance sampling weights
        weights = (len(self.buffer) * prob_distribution[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Increase beta over time for more accurate expected value approximation
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled experiences based on TD errors.
        
        Args:
            indices: Indices of sampled experiences
            priorities: New priorities (typically absolute TD errors + epsilon)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = (priority + self.epsilon) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_full(self):
        """Check if buffer has reached capacity."""
        return len(self.buffer) == self.capacity


class SegmentTree:
    """Segment tree data structure for efficient priority sampling.
    
    A more efficient implementation of prioritized replay using segment trees
    for sum and min operations with O(log n) complexity.
    """
    
    def __init__(self, capacity, operation, neutral_element):
        """Initialize Segment Tree.
        
        Args:
            capacity: Max size of tree (should be a power of 2)
            operation: Function to merge nodes (e.g., min or sum)
            neutral_element: Identity element for the operation (e.g., inf for min, 0 for sum)
        """
        # Ensure capacity is a power of 2 for a complete tree
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
            
        self.tree = [neutral_element] * (2 * self.capacity - 1)
        self.operation = operation
        self.neutral_element = neutral_element
        
    def _propagate(self, idx):
        """Propagate change in leaf node up through tree."""
        parent = (idx - 1) // 2
        
        while parent >= 0:
            self.tree[parent] = self.operation(
                self.tree[parent * 2 + 1],
                self.tree[parent * 2 + 2]
            )
            parent = (parent - 1) // 2
            
    def _retrieve(self, idx, value):
        """Find the highest index such that sum(tree[:idx]) <= value."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])
            
    def __setitem__(self, idx, val):
        """Set value at leaf node and update tree."""
        idx += self.capacity - 1  # Convert to leaf index
        
        self.tree[idx] = val
        self._propagate(idx)
        
    def __getitem__(self, idx):
        """Get value at leaf node."""
        idx += self.capacity - 1  # Convert to leaf index
        return self.tree[idx]
    
    def get_prefix_sum_idx(self, prefix_sum):
        """Find highest idx such that sum(tree[:idx]) <= prefix_sum."""
        return self._retrieve(0, prefix_sum)
    
    def sum(self):
        """Return sum of all values in tree."""
        return self.tree[0]
    
    def min(self):
        """Return minimum value in tree."""
        return self.tree[0]


class SumSegmentTree(SegmentTree):
    """Sum segment tree for prioritized replay buffer."""
    
    def __init__(self, capacity):
        """Initialize with sum operation and neutral element 0."""
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )
        
    def sum(self, start=0, end=None):
        """Calculate sum of elements in range [start, end)."""
        if end is None:
            end = self.capacity
        return super().sum()
    
    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index such that the sum of elements up to that index <= prefixsum."""
        assert 0 <= prefixsum <= self.sum() + 1e-5
        return self.get_prefix_sum_idx(prefixsum)


class MinSegmentTree(SegmentTree):
    """Min segment tree for prioritized replay buffer."""
    
    def __init__(self, capacity):
        """Initialize with min operation and neutral element infinity."""
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )
        
    def min(self, start=0, end=None):
        """Calculate min of elements in range [start, end)."""
        if end is None:
            end = self.capacity
        return super().min()


class EfficientPrioritizedReplayBuffer:
    """More efficient prioritized replay buffer using segment trees."""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        """Initialize efficient prioritized replay buffer.
        
        Args:
            capacity: Maximum size of the replay buffer
            alpha: Exponent determining how much prioritization is used (0=no prioritization, 1=full prioritization)
            beta: Importance-sampling correction exponent
            beta_increment: Small value to increase beta over time towards 1
            epsilon: Small constant added to priorities to ensure non-zero probability
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0
        
        # Buffer to store experiences
        self.buffer = []
        self.position = 0
        
        # Segment trees for efficient priority operations
        self.sum_tree = SumSegmentTree(capacity)
        self.min_tree = MinSegmentTree(capacity)
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory with max priority.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        # New experiences get max priority
        priority = self.max_priority ** self.alpha
        self.sum_tree[self.position] = priority
        self.min_tree[self.position] = priority
        
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample a batch of experiences based on priorities using segment trees.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple containing:
            - Batched states, actions, rewards, next_states, and dones
            - Indices of sampled experiences
            - Importance sampling weights
        """
        assert len(self.buffer) > 0, "Cannot sample from an empty buffer"
        
        indices = []
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Calculate segment size
        segment_size = self.sum_tree.sum() / batch_size
        
        # Increase beta for more accurate importance sampling weights over time
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Minimum priority for normalization of weights
        min_priority = self.min_tree.min() / self.sum_tree.sum()
        
        # Sample from each segment
        for i in range(batch_size):
            # Uniformly sample from segment
            a = segment_size * i
            b = segment_size * (i + 1)
            mass = random.uniform(a, b)
            
            # Retrieve sample from tree using priority
            idx = self.sum_tree.find_prefixsum_idx(mass)
            indices.append(idx)
            
            # Calculate importance sampling weight
            priority = self.sum_tree[idx] / self.sum_tree.sum()
            weights[i] = (priority / min_priority) ** (-self.beta)
        
        # Normalize weights
        weights = weights / weights.max()
        
        # Get experiences for sampled indices
        experiences = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities in segment trees.
        
        Args:
            indices: Indices of sampled experiences
            priorities: New priorities (typically absolute TD errors + epsilon)
        """
        for idx, priority in zip(indices, priorities):
            # Add epsilon to ensure non-zero probability
            priority = (priority + self.epsilon) ** self.alpha
            
            # Update segment trees
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)
    
    def is_full(self):
        """Check if buffer has reached capacity."""
        return len(self.buffer) == self.capacity