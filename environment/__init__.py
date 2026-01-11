"""Register CollaborativeCarry environment with EPyMARL."""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from environment.epymarl_wrapper import CollaborativeCarryMARL


REGISTRY = {}


def register_env(name, env_class):
    REGISTRY[name] = env_class


register_env("collaborative_carry", CollaborativeCarryMARL)


def make_env(env_name, **kwargs):
    """Factory function to create environment instance."""
    if env_name not in REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(REGISTRY.keys())}")
    return REGISTRY[env_name](**kwargs)
