"""
Deep Q-Network (DQN) Agent Implementation

This module implements a DQN agent with the following features:
- Experience replay memory to break correlations between sequential samples
- Separate target network for stable learning
- Double DQN implementation to reduce overestimation bias
- Epsilon-greedy exploration strategy with annealing
- Various optimization parameters and hyperparameters for tuning
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import time
from collections import deque

# Local imports
from dqn_model import DQNModel, DuelingDQNModel
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """Deep Q-Network agent that learns from environment interactions."""
    
    def __init__(self, state_dim, action_dim, config=None):
        """Initialize DQN agent with given state and action dimensions.
        
        Args:
            state_dim: Dimensions of the state space (can be int or tuple)
            action_dim: Number of possible actions
            config: Dictionary with hyperparameters
        """
        # Default configuration
        self.config = {
            # Model parameters
            'hidden_dim': 128,
            'learning_rate': 0.0005,
            'gamma': 0.99,  # Discount factor
            
            # Training parameters
            'batch_size': 64,
            'target_update_freq': 1000,  # Update target network every N steps
            'update_freq': 4,  # Update network every N steps
            
            # Exploration parameters
            'epsilon_start': 1.0,
            'epsilon_end': 0.1,
            'epsilon_decay': 10000,  # Steps for epsilon decay
            
            # Memory parameters
            'memory_size': 100000,
            'prioritized_replay': False,
            'alpha': 0.6,  # PER - How much prioritization to use (0=none, 1=full)
            'beta': 0.4,  # PER - Importance sampling correction
            'beta_increment': 0.001,  # PER - Increment for beta over time
            
            # Advanced options
            'double_dqn': True,
            'dueling': False,
            'clip_grad': 10.0,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        # Update config with provided values
        if config:
            self.config.update(config)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(self.config['device'])
        
        # Initialize networks
        if self.config['dueling']:
            self.policy_net = DuelingDQNModel(state_dim, action_dim, self.config['hidden_dim']).to(self.device)
            self.target_net = DuelingDQNModel(state_dim, action_dim, self.config['hidden_dim']).to(self.device)
        else:
            self.policy_net = DQNModel(state_dim, action_dim, self.config['hidden_dim']).to(self.device)
            self.target_net = DQNModel(state_dim, action_dim, self.config['hidden_dim']).to(self.device)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is only used for inference
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config['learning_rate'])
        
        # Initialize replay memory
        if self.config['prioritized_replay']:
            self.memory = PrioritizedReplayBuffer(
                self.config['memory_size'],
                self.config['alpha'],
                self.config['beta'],
                self.config['beta_increment']
            )
        else:
            self.memory = ReplayBuffer(self.config['memory_size'])
        
        # Initialize exploration parameters
        self.epsilon = self.config['epsilon_start']
        
        # Training metrics
        self.steps_done = 0
        self.episodes_done = 0
        self.total_rewards = deque(maxlen=100)  # Store last 100 episode rewards
        self.training_start_time = time.time()
        
    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether to use exploration (True) or exploitation only (False)
            
        Returns:
            Selected action index
        """
        # Anneal epsilon
        if training:
            eps_threshold = self.config['epsilon_end'] + (self.config['epsilon_start'] - self.config['epsilon_end']) * \
                            np.exp(-1. * self.steps_done / self.config['epsilon_decay'])
            self.epsilon = eps_threshold
            self.steps_done += 1
        else:
            eps_threshold = 0.0  # No exploration during evaluation
        
        # Epsilon-greedy action selection
        if training and random.random() < eps_threshold:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                # Add batch dimension and convert to tensor
                if isinstance(state, np.ndarray):
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.policy_net(state).argmax(dim=1).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
        """
        self.memory.add(state, action, reward, next_state, done)
    
    def update(self):
        """Update policy network parameters using a batch from replay memory."""
        if len(self.memory) < self.config['batch_size']:
            return 0.0  # Not enough samples for an update
        
        # Sample batch from replay memory
        if self.config['prioritized_replay']:
            experiences, indices, weights = self.memory.sample(self.config['batch_size'])
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = self.memory.sample(self.config['batch_size'])
            weights = torch.ones(self.config['batch_size']).to(self.device)
        
        # Unpack experiences
        states, actions, rewards, next_states, dones = experiences
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute next Q values using target network
        with torch.no_grad():
            if self.config['double_dqn']:
                # Double DQN: use policy net to select action, target net to evaluate it
                next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # Regular DQN: use maximum Q value from target network
                next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
            
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * self.config['gamma'] * next_q_values
        
        # Compute loss with importance sampling weights for prioritized replay
        td_errors = target_q_values - current_q_values
        loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Update priorities in prioritized replay buffer
        if self.config['prioritized_replay']:
            new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
            self.memory.update_priorities(indices, new_priorities)
        
        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        if self.config['clip_grad'] > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config['clip_grad'])
            
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update target network with current policy network parameters."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def train(self, env, num_episodes, max_steps_per_episode=1000, 
              update_interval=1, render=False, render_interval=100):
        """Train the agent on the given environment.
        
        Args:
            env: Environment to train on (should follow gym interface)
            num_episodes: Number of episodes to train for
            max_steps_per_episode: Maximum steps per episode
            update_interval: How often to update the network (in steps)
            render: Whether to render the environment
            render_interval: How often to render episodes
            
        Returns:
            List of rewards per episode
        """
        episode_rewards = []
        
        for episode in range(num_episodes):
            self.episodes_done += 1
            state, _ = env.reset()
            episode_reward = 0
            
            # Convert initial state to appropriate format if necessary
            state = self._process_state(state)
            
            for step in range(max_steps_per_episode):
                # Render environment if required
                if render and episode % render_interval == 0:
                    env.render()
                
                # Select and execute action
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Process next state
                next_state = self._process_state(next_state)
                
                # Store transition in replay memory
                self.store_transition(state, action, reward, next_state, done)
                
                # Update policy network
                if self.steps_done % update_interval == 0:
                    loss = self.update()
                
                # Update target network periodically
                if self.steps_done % self.config['target_update_freq'] == 0:
                    self.update_target_network()
                
                # Update state and tracking variables
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Record episode reward
            episode_rewards.append(episode_reward)
            self.total_rewards.append(episode_reward)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.total_rewards) if self.total_rewards else 0
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Steps: {self.steps_done} | "
                      f"Epsilon: {self.epsilon:.3f} | "
                      f"Avg Reward: {avg_reward:.2f}")
        
        return episode_rewards
    
    def _process_state(self, state):
        """Process state into appropriate format for network input.
        
        Args:
            state: Raw state from environment
            
        Returns:
            Processed state suitable for network input
        """
        # Default implementation - override in subclasses if needed
        if isinstance(state, dict):
            # If state is a dictionary, assume it's a complex state and extract relevant parts
            # This is common in environments like gym where obs might be a dictionary
            return np.array(list(state.values())).flatten()
        return state
    
    def save(self, path):
        """Save model weights and agent state.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
        }, path)
        print(f"Agent saved to {path}")
    
    def load(self, path):
        """Load model weights and agent state.
        
        Args:
            path: Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load network parameters
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Load other agent state
        self.steps_done = checkpoint.get('steps_done', 0)
        self.episodes_done = checkpoint.get('episodes_done', 0)
        
        # Optionally update config
        saved_config = checkpoint.get('config', {})
        for key, value in saved_config.items():
            if key in self.config:
                self.config[key] = value
        
        print(f"Agent loaded from {path}")