"""Configuration settings for the HUMAN 2.0 system."""

from typing import Dict, List, Any
from pathlib import Path
import os
import json
import logging
import logging.config

# Base directories
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
LOGS_DIR = BASE_DIR / 'logs'

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / 'human.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

# Pattern Recognition Settings
PATTERN_RECOGNITION = {
    'max_history': 10000,
    'min_pattern_confidence': 0.7,
    'significance_level': 0.05,
    'window_sizes': [10, 30, 60, 300],
    'min_cycle_periods': [2, 5, 10, 30],
    'max_patterns_per_variable': 100,
    'causality_lag_max': 10,
    'min_causality_confidence': 0.6
}

# Emotional System Settings
EMOTIONAL_SYSTEM = {
    'memory_capacity': 1000,
    'learning_rate': 0.001,
    'emotion_decay_rate': 0.1,
    'min_emotion_threshold': 0.1,
    'max_emotion_intensity': 1.0,
    'emotion_update_interval': 60,  # seconds
    'base_emotions': ['joy', 'sadness', 'anger', 'fear', 'surprise']
}

# Learning System Settings
LEARNING_SYSTEM = {
    'max_training_samples': 10000,
    'batch_size': 32,
    'validation_split': 0.2,
    'early_stopping_patience': 5,
    'model_save_frequency': 100,  # iterations
    'min_improvement_threshold': 0.01
}

# Security Settings
SECURITY = {
    'max_self_modification_depth': 2,
    'allowed_modification_paths': ['src/components', 'src/models'],
    'forbidden_imports': ['os.system', 'subprocess', 'eval', 'exec'],
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'allowed_file_types': ['.py', '.json', '.txt', '.md']
}

# Resource Limits
RESOURCE_LIMITS = {
    'max_cpu_percent': 80,
    'max_memory_percent': 70,
    'max_disk_usage_percent': 90,
    'max_network_bandwidth': 1000000,  # bytes/sec
    'max_open_files': 1000
}

class Config:
    """Configuration class for HUMAN 2.0 system."""
    
    def __init__(self):
        self.base_dir = BASE_DIR
        self.data_dir = DATA_DIR
        self.models_dir = MODELS_DIR
        self.logs_dir = LOGS_DIR
        
        # Load all configuration sections
        self.pattern_recognition = PATTERN_RECOGNITION
        self.emotional_system = EMOTIONAL_SYSTEM
        self.learning_system = LEARNING_SYSTEM
        self.security = SECURITY
        self.resource_limits = RESOURCE_LIMITS
        
        # Load custom configuration
        self.custom_config = load_custom_config()
        if self.custom_config:
            self._update_from_custom()
    
    def _update_from_custom(self):
        """Update configuration from custom config."""
        update_config(self.custom_config)
        
        # Update instance attributes
        if 'PATTERN_RECOGNITION' in self.custom_config:
            self.pattern_recognition.update(self.custom_config['PATTERN_RECOGNITION'])
        if 'EMOTIONAL_SYSTEM' in self.custom_config:
            self.emotional_system.update(self.custom_config['EMOTIONAL_SYSTEM'])
        if 'LEARNING_SYSTEM' in self.custom_config:
            self.learning_system.update(self.custom_config['LEARNING_SYSTEM'])
        if 'SECURITY' in self.custom_config:
            self.security.update(self.custom_config['SECURITY'])
        if 'RESOURCE_LIMITS' in self.custom_config:
            self.resource_limits.update(self.custom_config['RESOURCE_LIMITS'])
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return getattr(self, key, default)
    
    def set(self, key: str, value):
        """Set configuration value."""
        setattr(self, key, value)

def load_custom_config() -> Dict[str, Any]:
    """Load custom configuration from config.json if it exists."""
    config_path = BASE_DIR / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            return custom_config
        except Exception as e:
            logging.error(f"Error loading custom config: {e}")
    return {}

def update_config(custom_config: Dict[str, Any]) -> None:
    """Update configuration with custom settings."""
    def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    # Update each config section
    sections = [
        'PATTERN_RECOGNITION',
        'EMOTIONAL_SYSTEM',
        'LEARNING_SYSTEM',
        'SECURITY',
        'RESOURCE_LIMITS'
    ]
    
    for section in sections:
        if section in custom_config:
            globals()[section] = deep_update(globals()[section], custom_config[section])

# Load custom configuration if available
custom_config = load_custom_config()
if custom_config:
    update_config(custom_config)

# Initialize logging
logging.config.dictConfig(LOGGING_CONFIG) 