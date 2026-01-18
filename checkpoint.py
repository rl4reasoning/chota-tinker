"""Checkpointing utilities for trajectory collection.

Provides save/resume functionality for long-running trajectory collection jobs.
Checkpoints are saved to: checkpoints/<YYYYMMDD_HHMMSS>/

Usage:
    from checkpoint import CheckpointManager, get_checkpoint_dir

    # Create checkpoint directory
    checkpoint_dir = get_checkpoint_dir()  # or use resume_from path
    
    # Create manager
    manager = CheckpointManager(checkpoint_dir, args_dict={...})
    
    # Save checkpoint
    manager.save(active_states_data, completed_states_data, current_round)
    
    # Load checkpoint
    data = manager.load()
"""

import os
import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


def get_checkpoint_dir(base_dir: str = "checkpoints") -> str:
    """Generate a checkpoint directory with current timestamp.
    
    Args:
        base_dir: Base directory for checkpoints (default: "checkpoints")
        
    Returns:
        Path like "checkpoints/20260117_143052"
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, timestamp)


@dataclass
class CheckpointData:
    """Checkpoint data container.
    
    Attributes:
        active_states_data: List of serialized active rollout state dictionaries
        completed_states_data: List of serialized completed state dictionaries
        current_round: Current generation round (0-indexed, represents completed rounds)
        total_rounds: Total number of rounds planned
        args_dict: Run configuration for verification on resume
        timestamp: ISO format timestamp of checkpoint creation
        extra_data: Optional dictionary for additional data specific to the collector
    """
    active_states_data: list[dict]
    completed_states_data: list[dict]
    current_round: int
    total_rounds: int
    args_dict: dict
    timestamp: str
    extra_data: Optional[dict] = None


class CheckpointManager:
    """Manages checkpoint save/load operations.
    
    Example:
        manager = CheckpointManager("checkpoints/20260117_143052", args_dict={...})
        
        # During training loop:
        for round in range(start_round, total_rounds):
            # ... do generation ...
            manager.save(active_data, completed_data, round + 1, total_rounds)
        
        # To resume:
        if manager.has_checkpoint():
            data = manager.load()
            start_round = data.current_round
    """
    
    def __init__(self, checkpoint_dir: str, args_dict: Optional[dict] = None):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save/load checkpoints
            args_dict: Run configuration (saved with checkpoint for verification)
        """
        self.checkpoint_dir = checkpoint_dir
        self.args_dict = args_dict or {}
        self.checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pkl")
        self.info_path = os.path.join(checkpoint_dir, "checkpoint_info.json")
    
    def has_checkpoint(self) -> bool:
        """Check if a checkpoint exists."""
        return os.path.exists(self.checkpoint_path)
    
    def save(
        self,
        active_states_data: list[dict],
        completed_states_data: list[dict],
        current_round: int,
        total_rounds: int,
        extra_data: Optional[dict] = None,
    ) -> None:
        """Save checkpoint to disk.
        
        Args:
            active_states_data: Serialized active states
            completed_states_data: Serialized completed states
            current_round: Number of completed rounds
            total_rounds: Total rounds planned
            extra_data: Optional additional data to save
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        checkpoint_data = CheckpointData(
            active_states_data=active_states_data,
            completed_states_data=completed_states_data,
            current_round=current_round,
            total_rounds=total_rounds,
            args_dict=self.args_dict,
            timestamp=datetime.now().isoformat(),
            extra_data=extra_data,
        )
        
        # Save pickle checkpoint
        with open(self.checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        
        # Save human-readable info
        info = {
            "current_round": current_round,
            "total_rounds": total_rounds,
            "num_active_states": len(active_states_data),
            "num_completed_states": len(completed_states_data),
            "timestamp": checkpoint_data.timestamp,
            "args": self.args_dict,
        }
        if extra_data:
            info["extra"] = {k: str(v) if not isinstance(v, (int, float, bool, str, list)) else v 
                           for k, v in extra_data.items()}
        
        with open(self.info_path, "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"  Checkpoint saved to: {self.checkpoint_dir}")
    
    def load(self) -> CheckpointData:
        """Load checkpoint from disk.
        
        Returns:
            CheckpointData with saved state
            
        Raises:
            FileNotFoundError: If no checkpoint exists
        """
        if not self.has_checkpoint():
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path}")
        
        with open(self.checkpoint_path, "rb") as f:
            return pickle.load(f)
    
    def verify_args(self, current_args: dict, warn_keys: Optional[list[str]] = None) -> list[str]:
        """Verify current args match checkpoint args.
        
        Args:
            current_args: Current run configuration
            warn_keys: Keys to check and warn about if different
            
        Returns:
            List of warning messages for mismatched keys
        """
        if not self.has_checkpoint():
            return []
        
        data = self.load()
        saved_args = data.args_dict
        warnings = []
        
        warn_keys = warn_keys or ["num_problems", "num_samples", "dataset", "model"]
        
        for key in warn_keys:
            saved_val = saved_args.get(key)
            current_val = current_args.get(key)
            if saved_val != current_val:
                warnings.append(f"  Warning: {key} changed ({saved_val} -> {current_val})")
        
        return warnings
    
    def get_info(self) -> Optional[dict]:
        """Get checkpoint info without loading full checkpoint.
        
        Returns:
            Info dict or None if no checkpoint exists
        """
        if not os.path.exists(self.info_path):
            return None
        
        with open(self.info_path, "r") as f:
            return json.load(f)
