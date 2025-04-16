"""
Path handling utilities for the img2latex project.

This module provides a consistent way to access project paths across
different modules and environments.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from img2latex.utils.logging import get_logger

logger = get_logger(__name__)


class PathManager:
    """
    Manages paths for the img2latex project.

    This class provides a centralized way to access project paths,
    ensuring consistency across different modules and environments.
    """

    def __init__(self, root_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the PathManager with project root directory.

        Args:
            root_dir: Path to the project root directory. If None, attempts to
                     determine the root directory automatically.
        """
        if root_dir is None:
            # Try to find the project root directory automatically
            current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
            self.root_dir = current_dir.parent.parent.parent  # utils -> img2latex -> project root

            # Validate we have the correct root dir by checking for expected directories
            if not (self.root_dir / "img2latex").exists():
                raise ValueError(
                    "Could not automatically determine project root directory. "
                    "Please provide root_dir manually."
                )
        else:
            self.root_dir = Path(root_dir)

        # Define standard paths
        self.img2latex_dir = self.root_dir / "img2latex"
        self.configs_dir = self.img2latex_dir / "configs"
        self.data_dir = self.root_dir / "data"
        self.outputs_dir = self.root_dir / "outputs"
        self.registry_dir = self.outputs_dir / "registry"

        # Create registry directory
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        # Registry file path
        self.experiment_registry_file = self.registry_dir / "experiment_registry.json"

    def get_experiment_dir(self, experiment_name: str) -> Path:
        """
        Get the directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment directory
        """
        # Check if experiment name contains a version number
        if "_v" in experiment_name:
            # Extract model name and version (e.g., "model_v2" -> "model" and "2")
            dir_name = experiment_name
        else:
            # If no version in name, use the experiment name directly with v1
            dir_name = f"{experiment_name}_v1"

        experiment_dir = self.outputs_dir / dir_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return experiment_dir

    def get_checkpoint_dir(self, experiment_name: str) -> Path:
        """
        Get the checkpoints directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's checkpoint directory
        """
        checkpoint_dir = self.get_experiment_dir(experiment_name) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def get_log_dir(self, experiment_name: str) -> Path:
        """
        Get the logs directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's log directory
        """
        log_dir = self.get_experiment_dir(experiment_name) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def get_metrics_dir(self, experiment_name: str) -> Path:
        """
        Get the metrics directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's metrics directory
        """
        metrics_dir = self.get_experiment_dir(experiment_name) / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        return metrics_dir

    def get_reports_dir(self, experiment_name: str) -> Path:
        """
        Get the reports directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's reports directory
        """
        reports_dir = self.get_experiment_dir(experiment_name) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        return reports_dir

    def get_plots_dir(self, experiment_name: str) -> Path:
        """
        Get the plots directory for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's plots directory
        """
        plots_dir = self.get_experiment_dir(experiment_name) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        return plots_dir

    def get_config_path(self, experiment_name: str) -> Path:
        """
        Get the path to the config.yaml file for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to the experiment's config.yaml file
        """
        return self.get_experiment_dir(experiment_name) / "config.yaml"

    def register_experiment(self, experiment_name: str, metadata: Dict = None) -> None:
        """
        Register a new experiment in the experiment registry.

        Args:
            experiment_name: Name of the experiment
            metadata: Additional metadata to store for the experiment
        """
        registry = self._load_registry()

        # Add or update experiment entry
        if experiment_name not in registry:
            registry[experiment_name] = {
                "creation_time": self._get_timestamp(),
                "path": str(self.get_experiment_dir(experiment_name)),
                "metadata": metadata or {},
            }
        else:
            # Update existing entry with new metadata
            registry[experiment_name]["last_updated"] = self._get_timestamp()
            if metadata:
                registry[experiment_name]["metadata"].update(metadata)

        # Save updated registry
        self._save_registry(registry)
        logger.info(f"Registered experiment: {experiment_name}")

    def get_experiment_metadata(self, experiment_name: str) -> Dict:
        """
        Get metadata for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary of experiment metadata
        """
        registry = self._load_registry()
        if experiment_name in registry:
            return registry[experiment_name]
        return {}

    def list_experiments(self) -> List[str]:
        """
        Get a list of all registered experiments.

        Returns:
            List of experiment names
        """
        registry = self._load_registry()
        return list(registry.keys())

    def _load_registry(self) -> Dict:
        """
        Load the experiment registry from disk.

        Returns:
            Dictionary containing the experiment registry
        """
        if not self.experiment_registry_file.exists():
            return {}

        try:
            with open(self.experiment_registry_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading experiment registry: {e}")
            return {}

    def _save_registry(self, registry: Dict) -> None:
        """
        Save the experiment registry to disk.

        Args:
            registry: Dictionary containing the experiment registry
        """
        try:
            with open(self.experiment_registry_file, "w") as f:
                json.dump(registry, f, indent=2, sort_keys=True)
        except Exception as e:
            logger.error(f"Error saving experiment registry: {e}")

    def _get_timestamp(self) -> str:
        """
        Get the current timestamp as a string.

        Returns:
            Current timestamp as a string
        """
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def create_experiment_structure(self, experiment_name: str) -> Dict[str, Path]:
        """
        Create the complete directory structure for an experiment.

        This method creates all the subdirectories for an experiment and
        returns a dictionary with the paths.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary with path names as keys and Path objects as values
        """
        # Create main experiment directory
        experiment_dir = self.get_experiment_dir(experiment_name)

        # Create all subdirectories
        paths = {
            "experiment_dir": experiment_dir,
            "checkpoints_dir": self.get_checkpoint_dir(experiment_name),
            "logs_dir": self.get_log_dir(experiment_name),
            "metrics_dir": self.get_metrics_dir(experiment_name),
            "reports_dir": self.get_reports_dir(experiment_name),
            "plots_dir": self.get_plots_dir(experiment_name),
            "config_path": self.get_config_path(experiment_name),
        }

        # Register the experiment
        self.register_experiment(experiment_name)

        logger.info(f"Created directory structure for experiment: {experiment_name}")
        return paths

    def as_dict(self) -> Dict[str, Path]:
        """
        Get all project paths as a dictionary.

        Returns:
            Dictionary with path names as keys and Path objects as values
        """
        return {
            "root_dir": self.root_dir,
            "img2latex_dir": self.img2latex_dir,
            "configs_dir": self.configs_dir,
            "data_dir": self.data_dir,
            "outputs_dir": self.outputs_dir,
            "registry_dir": self.registry_dir,
        }


# Create a singleton instance for easy import
path_manager = PathManager()