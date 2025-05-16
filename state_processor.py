"""
State Processor Module for DRL Agents

This module provides functionality for preprocessing environment state observations
into formats suitable for use with deep reinforcement learning agents.
Common preprocessing steps include:
- Resizing images
- Converting to grayscale
- Normalization
- Stacking frames for temporal information
- Feature extraction or embeddings
"""

import numpy as np
import cv2
from collections import deque


class StateProcessor:
    """Base class for state processing."""
    
    def __init__(self):
        """Initialize the state processor."""
        pass
    
    def process(self, state):
        """Process a state observation.
        
        Args:
            state: Raw state from environment
            
        Returns:
            Processed state suitable for agent input
        """
        # Base implementation is identity function
        return state


class ImageProcessor(StateProcessor):
    """Process image-based states (like Atari game screens)."""
    
    def __init__(self, height=84, width=84, grayscale=True, normalize=True):
        """Initialize image processor.
        
        Args:
            height: Target image height
            width: Target image width
            grayscale: Whether to convert to grayscale
            normalize: Whether to normalize pixel values to [0,1]
        """
        super(ImageProcessor, self).__init__()
        self.height = height
        self.width = width
        self.grayscale = grayscale
        self.normalize = normalize
        
    def process(self, frame):
        """Process a single frame.
        
        Args:
            frame: Raw frame from environment, typically a RGB image array
            
        Returns:
            Processed frame
        """
        if frame is None:
            return None
            
        # Convert to grayscale if needed
        if self.grayscale and frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize image
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
            
        # Ensure correct shape
        if self.grayscale:
            frame = frame.reshape(self.height, self.width, 1)
            
        return frame


class FrameStacker(StateProcessor):
    """Stack multiple frames to provide temporal information."""
    
    def __init__(self, num_frames=4, processor=None):
        """Initialize frame stacker.
        
        Args:
            num_frames: Number of frames to stack
            processor: Optional state processor to apply to each frame
        """
        super(FrameStacker, self).__init__()
        self.num_frames = num_frames
        self.processor = processor if processor else StateProcessor()
        self.frames = deque(maxlen=num_frames)
        self.initialized = False
        
    def reset(self):
        """Clear frame history."""
        self.frames.clear()
        self.initialized = False
        
    def process(self, frame):
        """Process and stack frames.
        
        Args:
            frame: New frame from environment
            
        Returns:
            Stacked frames as a single array
        """
        # Process the new frame
        processed_frame = self.processor.process(frame)
        
        # Initialize the stack if needed
        if not self.initialized:
            # Fill with copies of first frame
            for _ in range(self.num_frames):
                self.frames.append(processed_frame)
            self.initialized = True
        else:
            # Add new frame to stack
            self.frames.append(processed_frame)
            
        # Stack frames along a new axis
        return np.stack(self.frames, axis=0)


class VectorNormalizer(StateProcessor):
    """Normalize numerical state vectors."""
    
    def __init__(self, mean=None, std=None, low=None, high=None, 
                 clip_range=None, running_stats=False):
        """Initialize vector normalizer.
        
        Args:
            mean: Optional fixed mean for standardization
            std: Optional fixed standard deviation for standardization
            low: Optional fixed min values for min-max normalization
            high: Optional fixed max values for min-max normalization
            clip_range: Optional (min, max) to clip normalized values
            running_stats: Whether to compute stats online from observed states
        """
        super(VectorNormalizer, self).__init__()
        
        self.mean = mean
        self.std = std
        self.low = low
        self.high = high
        self.clip_range = clip_range
        self.running_stats = running_stats
        
        # For running statistics
        if running_stats:
            self.count = 0
            self.running_mean = 0
            self.running_var = 0
            self.running_min = float('inf')
            self.running_max = float('-inf')
            
    def update_stats(self, state):
        """Update running statistics with new state.
        
        Args:
            state: State vector
            
        Uses Welford's online algorithm for variance
        """
        self.count += 1
        
        # Update min and max
        self.running_min = np.minimum(self.running_min, state)
        self.running_max = np.maximum(self.running_max, state)
        
        # For first state, initialize mean and set delta to zero
        if self.count == 1:
            self.running_mean = state.astype(np.float64)
        else:
            old_mean = self.running_mean.copy()
            self.running_mean += (state - old_mean) / self.count
            self.running_var += (state - old_mean) * (state - self.running_mean)
    
    def process(self, state):
        """Normalize state vector.
        
        Args:
            state: Raw state vector from environment
            
        Returns:
            Normalized state vector
        """
        state = np.array(state, dtype=np.float32)
        
        # Update running statistics if needed
        if self.running_stats:
            self.update_stats(state)
        
        # Apply normalization based on available stats
        if self.mean is not None and self.std is not None:
            # Standardization (z-score normalization)
            normalized_state = (state - self.mean) / (self.std + 1e-8)
            
        elif self.low is not None and self.high is not None:
            # Min-max normalization to [0, 1]
            normalized_state = (state - self.low) / (self.high - self.low + 1e-8)
            
        elif self.running_stats and self.count > 1:
            # Use running statistics
            if any(self.running_max - self.running_min > 1e-8):
                # Min-max normalization if we have meaningful range
                normalized_state = (state - self.running_min) / (self.running_max - self.running_min + 1e-8)
            else:
                # Standardization if min-max range is too small
                std = np.sqrt(self.running_var / (self.count - 1) + 1e-8)
                normalized_state = (state - self.running_mean) / (std + 1e-8)
        else:
            # No normalization
            normalized_state = state
        
        # Clip if needed
        if self.clip_range is not None:
            normalized_state = np.clip(normalized_state, self.clip_range[0], self.clip_range[1])
            
        return normalized_state


class FeatureExtractor(StateProcessor):
    """Extract features from raw state observations."""
    
    def __init__(self, feature_fn=None, num_features=None):
        """Initialize feature extractor.
        
        Args:
            feature_fn: Function to extract features
            num_features: Expected number of features (for validation)
        """
        super(FeatureExtractor, self).__init__()
        self.feature_fn = feature_fn
        self.num_features = num_features
        
    def process(self, state):
        """Extract features from state.
        
        Args:
            state: Raw state from environment
            
        Returns:
            Feature vector
        """
        if self.feature_fn is None:
            return state
            
        features = self.feature_fn(state)
        
        # Validate feature vector
        if self.num_features is not None:
            assert len(features) == self.num_features, \
                f"Expected {self.num_features} features, got {len(features)}"
                
        return features


class CompositeProcessor(StateProcessor):
    """Apply a sequence of processors in order."""
    
    def __init__(self, processors):
        """Initialize composite processor.
        
        Args:
            processors: List of StateProcessor objects to apply in sequence
        """
        super(CompositeProcessor, self).__init__()
        self.processors = processors
        
    def reset(self):
        """Reset all processors that have reset method."""
        for processor in self.processors:
            if hasattr(processor, 'reset'):
                processor.reset()
                
    def process(self, state):
        """Apply all processors in sequence.
        
        Args:
            state: Raw state from environment
            
        Returns:
            Final processed state
        """
        processed_state = state
        for processor in self.processors:
            processed_state = processor.process(processed_state)
            
        return processed_state


class DictProcessor(StateProcessor):
    """Process states that are dictionaries of values."""
    
    def __init__(self, processors=None, keys=None, flatten=False):
        """Initialize dictionary processor.
        
        Args:
            processors: Dictionary mapping keys to processors
            keys: List of keys to extract (if None, use all keys)
            flatten: Whether to flatten the results into a single array
        """
        super(DictProcessor, self).__init__()
        self.processors = processors or {}
        self.keys = keys
        self.flatten = flatten
        
    def process(self, state):
        """Process a dictionary state.
        
        Args:
            state: Dictionary state from environment
            
        Returns:
            Processed state as dictionary or flattened array
        """
        if not isinstance(state, dict):
            raise ValueError("DictProcessor requires dictionary input")
            
        # Use all keys if none specified
        keys_to_process = self.keys or state.keys()
        
        processed_state = {}
        for key in keys_to_process:
            if key in state:
                # Get processor for this key or use identity
                processor = self.processors.get(key, StateProcessor())
                processed_state[key] = processor.process(state[key])
        
        # Flatten if requested
        if self.flatten:
            flattened = []
            for key in sorted(processed_state.keys()):
                value = processed_state[key]
                if isinstance(value, np.ndarray):
                    flattened.append(value.flatten())
                elif isinstance(value, (list, tuple)):
                    flattened.append(np.array(value).flatten())
                else:
                    flattened.append(np.array([value]).flatten())
            return np.concatenate(flattened)
            
        return processed_state