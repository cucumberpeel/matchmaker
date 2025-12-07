import json
import numpy as np
import gymnasium as gym
from feature_extractor import compute_features


class ValueMatchingEnv(gym.Env):

    def __init__(self, env_config):
        super(ValueMatchingEnv, self).__init__()
        self.primitives = env_config['primitives']
        self.costs = env_config["costs"]
        self.dataset_path = env_config['dataset']
        self.feature_dim = env_config['feature_dim']
        self.max_steps = env_config['max_steps']
        self.action_space = gym.spaces.Discrete(len(self.primitives))
        self.steps_taken = 0
        self.action_history = [-1] * self.max_steps

        # observation_space is used by some RL libraries (e.g. Ray) to initialize neural networks
        self.observation_space = gym.spaces.Box(
                low=0.0,
                high=np.inf,
                shape=(self.feature_dim + self.max_steps,),
                dtype=np.float32
            )
        
        self.dataset = None  # will load lazily
    
    def reset(self, *, seed=None, options=None):
        # Reset episode state
        self._load_dataset()
        idx = np.random.randint(len(self.dataset))
        self.source = self.dataset[idx]['source_value']
        self.targets = self.dataset[idx]['target_values']
        self.gold = self.dataset[idx]['gold_value']
        self.steps_taken = 0
        self.action_history = [-1] * self.max_steps
        
        # Compute initial state
        features = compute_features(self.source, self.targets)
        action_history_encoded = self._encode_history()
        state = features + action_history_encoded
        state = np.array(state, dtype=np.float32)
        self.features = features

        # Compute initial state
        self.features = compute_features(self.source, self.targets)
        state = self._build_observation()

        info = {}
        return state, info
    
    def step(self, action):
        # Record algorithm in history
        if self.steps_taken < self.max_steps:
            self.action_history[self.steps_taken] = action
        self.steps_taken += 1 

        if action in self.action_history[:self.steps_taken-1]:  # Check previous actions only
            # Penalize heavily for selecting an already-used action
            reward = -0.5
            done = True
            truncated = False
            next_state = self._build_observation()
            
            info = {
                'predicted': None,
                'correct': False,
                'attempts': self.steps_taken,
                'history': self.action_history,
                'invalid_action': True
            }
            
            return next_state, reward, done, truncated, info

        # For LLM action, assume worst-case (wrong) prediction instead of
        # the previous oracle behaviour that returned the gold label.
        # This prevents the agent from learning to always pick the LLM.
        if action == 2:  # LLM reasoning
            predicted = None
        else:
            predicted = self.primitives[action](self.source, self.targets)

        # Check if correct
        is_correct = (predicted == self.gold)
        action_cost = self.costs[action]
        # Compute reward with decay based on attempts
        if is_correct:
            reward = 1.0 - 0.2 * (self.steps_taken - 1)  # Decay: 1.0, 0.8, 0.6
            reward -= action_cost
            done = True
        elif self.steps_taken >= self.max_steps:
            reward = -1.0  # Failed after max attempts
            done = True
        else:
            reward = -0.1  # Small penalty for wrong attempt
            reward -= action_cost
            done = False

        next_state = self._build_observation()
        truncated = False

        info = {
            'predicted': predicted,
            'correct': is_correct,
            'attempts': self.steps_taken,
            'history': self.action_history
        }

        return next_state, reward, done, truncated, info
    
    def _encode_history(self):
        """Normalize action history to [0, 1] range for better working with NN"""
        encoded = []
        for action in self.action_history:
            if action == -1:
                encoded.append(0.0)
            else:
                encoded.append((action + 1) / len(self.primitives))

        return encoded
    
    def _build_observation(self):
        """Build the observation vector"""
        action_history_encoded = self._encode_history()
        obs = self.features + action_history_encoded
        
        return np.array(obs, dtype=np.float32)
    
    def _load_dataset(self):
        if self.dataset is not None:
            return

        if isinstance(self.dataset_path, list):
            self.dataset = self.dataset_path
            return

        if isinstance(self.dataset_path, str):
            with open(self.dataset_path, "r") as f:
                self.dataset = json.load(f)
            return