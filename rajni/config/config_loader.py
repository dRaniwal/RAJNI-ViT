"""
Configuration loading and management utilities.
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "vit_base_patch16_224"
    pretrained: bool = True
    num_classes: int = 1000
    image_size: int = 224


@dataclass
class PruningConfig:
    """RAJNI pruning configuration."""
    pruning_ratio: float = 0.3
    num_pruning_layers: int = 6
    keep_cls_token: bool = True
    importance_metric: str = "jacobian"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    batch_size: int = 32
    num_workers: int = 4
    dataset: str = "imagenet"
    data_path: str = "./data"
    device: str = "cuda"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_dir: str = "./logs"
    log_level: str = "INFO"
    save_predictions: bool = False


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment_name: str = "rajni_eval"
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        config: Configuration object
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    # Parse nested configs
    model_config = ModelConfig(**config_dict.get('model', {}))
    pruning_config = PruningConfig(**config_dict.get('pruning', {}))
    eval_config = EvalConfig(**config_dict.get('eval', {}))
    logging_config = LoggingConfig(**config_dict.get('logging', {}))
    
    config = Config(
        model=model_config,
        pruning=pruning_config,
        eval=eval_config,
        logging=logging_config,
        experiment_name=config_dict.get('experiment_name', 'rajni_eval'),
        seed=config_dict.get('seed', 42)
    )
    
    return config


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object
        save_path: Path to save YAML file
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    with open(save_file, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """
    Merge base config with override dictionary.
    
    Args:
        base_config: Base configuration
        override_dict: Dictionary with override values
        
    Returns:
        merged_config: Merged configuration
    """
    base_dict = base_config.to_dict()
    
    # Deep merge
    def deep_update(base: dict, override: dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_update(base[key], value)
            else:
                base[key] = value
    
    deep_update(base_dict, override_dict)
    
    # Reconstruct config from merged dict
    model_config = ModelConfig(**base_dict.get('model', {}))
    pruning_config = PruningConfig(**base_dict.get('pruning', {}))
    eval_config = EvalConfig(**base_dict.get('eval', {}))
    logging_config = LoggingConfig(**base_dict.get('logging', {}))
    
    merged_config = Config(
        model=model_config,
        pruning=pruning_config,
        eval=eval_config,
        logging=logging_config,
        experiment_name=base_dict.get('experiment_name', 'rajni_eval'),
        seed=base_dict.get('seed', 42)
    )
    
    return merged_config
