import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from src.components.code_env import CodeOptimizationEnv
from src.embedder.code_embedder import CodeEmbedder

class TrainingMetricsCallback(BaseCallback):
    """Custom callback for collecting training metrics."""
    
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.metrics_file = self.log_dir / "training_metrics.jsonl"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
    def _on_step(self) -> bool:
        """Called after each step."""
        metrics = {
            'timestep': self.num_timesteps,
            'reward': float(self.locals['rewards'][0]),
            'episode_length': int(self.locals['episode_length']),
            'learning_rate': float(self.locals['self'].learning_rate),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
            
        return True

def make_env(
    code: str,
    rank: int,
    seed: int = 0,
    log_dir: Optional[str] = None
) -> callable:
    """Create a training environment."""
    def _init() -> gym.Env:
        env = CodeOptimizationEnv(code)
        if log_dir:
            env = Monitor(env, os.path.join(log_dir, str(rank)))
        env.seed(seed + rank)
        return env
    return _init

class CodeOptimizationTrainer:
    """Trainer for the code optimization model."""
    
    def __init__(
        self,
        base_dir: str,
        n_envs: int = 4,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        device: str = 'auto'
    ):
        self.base_dir = Path(base_dir)
        self.n_envs = n_envs
        self.device = device
        
        # Create directories
        self.log_dir = self.base_dir / "logs"
        self.model_dir = self.base_dir / "models"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize code embedder
        self.code_embedder = CodeEmbedder()
        
        # Training parameters
        self.training_params = {
            'learning_rate': learning_rate,
            'n_steps': n_steps,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma
        }
        
    def create_envs(self, code_samples: List[str]) -> DummyVecEnv:
        """Create vectorized environments for training."""
        env_fns = [
            make_env(
                code=code,
                rank=i,
                log_dir=str(self.log_dir)
            )
            for i, code in enumerate(code_samples[:self.n_envs])
        ]
        
        return DummyVecEnv(env_fns)
        
    def train(
        self,
        code_samples: List[str],
        total_timesteps: int = 1_000_000,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = 10000
    ) -> PPO:
        """Train the model on code samples."""
        # Create training environments
        env = self.create_envs(code_samples)
        
        # Create evaluation environment
        eval_env = Monitor(
            CodeOptimizationEnv(code_samples[0]),
            str(self.log_dir / "eval")
        )
        
        # Initialize model
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(self.log_dir),
            device=self.device,
            **self.training_params
        )
        
        # Setup callbacks
        metrics_callback = TrainingMetricsCallback(str(self.log_dir))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.model_dir / "best_model"),
            log_path=str(self.log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )
        
        # Train model
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=[metrics_callback, eval_callback],
                progress_bar=True
            )
            
            # Save final model
            final_model_path = self.model_dir / "final_model.zip"
            model.save(str(final_model_path))
            self.logger.info(f"Final model saved to {final_model_path}")
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
            
        return model
        
    def evaluate(
        self,
        model: PPO,
        code_samples: List[str],
        n_episodes: int = 10
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        eval_env = Monitor(
            CodeOptimizationEnv(code_samples[0]),
            str(self.log_dir / "final_eval")
        )
        
        # Run evaluation episodes
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
        # Calculate metrics
        metrics = {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_length': float(np.mean(episode_lengths)),
            'std_length': float(np.std(episode_lengths))
        }
        
        return metrics
        
    def save_training_config(self):
        """Save training configuration."""
        config = {
            'n_envs': self.n_envs,
            'device': self.device,
            'training_params': self.training_params,
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = self.log_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
if __name__ == "__main__":
    # Example usage
    trainer = CodeOptimizationTrainer("training_runs/run1")
    
    # Sample code for training
    code_samples = [
        '''
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total
        ''',
        '''
def process_data(data):
    results = []
    for item in data:
        results.append(item * 2)
    return results
        '''
    ]
    
    # Train model
    model = trainer.train(code_samples)
    
    # Evaluate model
    metrics = trainer.evaluate(model, code_samples)
    print("Evaluation metrics:", metrics) 