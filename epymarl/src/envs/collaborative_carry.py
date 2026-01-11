"""CollaborativeCarry environment for EPyMARL."""
import sys
from pathlib import Path

# Add miniRL to path
miniRL_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(miniRL_path))

from environment.epymarl_wrapper import CollaborativeCarryMARL

# Export for EPyMARL registry
def env_fn(**kwargs):
    return CollaborativeCarryMARL(**kwargs)
