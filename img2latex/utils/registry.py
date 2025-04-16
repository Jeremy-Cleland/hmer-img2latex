"""
Experiment registry for the img2latex project.

This module provides functionality for tracking, comparing, and managing experiments.
It builds on the PathManager's basic registry features to provide more advanced
experiment tracking capabilities.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from img2latex.utils.logging import get_logger
from img2latex.utils.path_utils import path_manager

logger = get_logger(__name__)


class ExperimentRegistry:
    """
    Advanced experiment registry for tracking and comparing experiments.

    This class extends the basic functionality in PathManager to provide
    comprehensive experiment tracking features.
    """

    def __init__(self):
        """Initialize the experiment registry."""
        self.path_manager = path_manager
        self.registry_dir = self.path_manager.registry_dir
        self.registry_file = self.path_manager.experiment_registry_file

        # Ensure the registry directory exists
        os.makedirs(self.registry_dir, exist_ok=True)

    def register_experiment(
        self,
        experiment_name: str,
        config: Optional[Dict] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> str:
        """
        Register a new experiment or update an existing one.

        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary used for the experiment
            description: Optional description of the experiment
            tags: Optional list of tags for categorizing experiments

        Returns:
            The experiment name with version (e.g., "experiment_v1")
        """
        # Get current registry data
        registry = self._load_registry()

        # Normalize the experiment name and get the versioned name
        if "_v" not in experiment_name:
            # Find the next version number if experiment exists
            base_name = experiment_name
            version = 1

            # Check if experiment name already exists with any version
            existing_versions = []
            for name in registry.keys():
                if name.startswith(f"{base_name}_v"):
                    try:
                        existing_versions.append(int(name.split("_v")[1]))
                    except ValueError:
                        continue

            if existing_versions:
                version = max(existing_versions) + 1

            experiment_name = f"{base_name}_v{version}"

        # Create experiment directory structure
        paths = self.path_manager.create_experiment_structure(experiment_name)

        # Prepare metadata
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = {
            "creation_time": timestamp,
            "last_updated": timestamp,
            "path": str(paths["experiment_dir"]),
            "description": description,
            "tags": tags or [],
            "metrics": {},
            "status": "created",
        }

        # Save configuration if provided
        if config:
            config_path = paths["config_path"]
            self._save_config(config_path, config)
            metadata["config_path"] = str(config_path)

        # Update registry
        registry[experiment_name] = metadata
        self._save_registry(registry)

        logger.info(f"Registered experiment: {experiment_name}")
        return experiment_name

    def update_experiment_status(self, experiment_name: str, status: str) -> None:
        """
        Update the status of an experiment.

        Args:
            experiment_name: Name of the experiment
            status: New status (e.g., 'running', 'completed', 'failed')
        """
        registry = self._load_registry()
        if experiment_name in registry:
            registry[experiment_name]["status"] = status
            registry[experiment_name]["last_updated"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            self._save_registry(registry)
            logger.info(f"Updated status of experiment {experiment_name} to '{status}'")
        else:
            logger.warning(
                f"Cannot update status: Experiment {experiment_name} not found in registry"
            )

    def log_metrics(
        self, experiment_name: str, metrics: Dict[str, Any], step: Optional[int] = None
    ) -> None:
        """
        Log metrics for an experiment.

        Args:
            experiment_name: Name of the experiment
            metrics: Dictionary of metrics to log
            step: Optional step number (e.g., epoch or iteration)
        """
        registry = self._load_registry()
        if experiment_name not in registry:
            logger.warning(
                f"Cannot log metrics: Experiment {experiment_name} not found in registry"
            )
            return

        # Update registry with metrics
        if "metrics" not in registry[experiment_name]:
            registry[experiment_name]["metrics"] = {}

        # If step is provided, organize metrics by step
        if step is not None:
            if "steps" not in registry[experiment_name]["metrics"]:
                registry[experiment_name]["metrics"]["steps"] = {}

            step_key = str(step)  # Convert to string for JSON compatibility
            if step_key not in registry[experiment_name]["metrics"]["steps"]:
                registry[experiment_name]["metrics"]["steps"][step_key] = {}

            # Update with new metrics
            registry[experiment_name]["metrics"]["steps"][step_key].update(metrics)
        else:
            # Just update the latest metrics (overwrite previous values)
            registry[experiment_name]["metrics"].update(metrics)

        # Update timestamp
        registry[experiment_name]["last_updated"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Save registry
        self._save_registry(registry)

        # Also save metrics to a separate file for the experiment
        metrics_dir = self.path_manager.get_metrics_dir(experiment_name)
        metrics_file = metrics_dir / "metrics.json"

        # Load existing metrics if file exists
        existing_metrics = {}
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    existing_metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing metrics: {e}")

        # Update with new metrics
        if step is not None:
            if "steps" not in existing_metrics:
                existing_metrics["steps"] = {}

            step_key = str(step)
            if step_key not in existing_metrics["steps"]:
                existing_metrics["steps"][step_key] = {}

            existing_metrics["steps"][step_key].update(metrics)
        else:
            existing_metrics.update(metrics)

        # Save updated metrics
        try:
            with open(metrics_file, "w") as f:
                json.dump(existing_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def get_experiment(self, experiment_name: str) -> Optional[Dict]:
        """
        Get details for a specific experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary with experiment details or None if not found
        """
        registry = self._load_registry()
        if experiment_name in registry:
            return registry[experiment_name]

        # Check if the experiment name is missing version
        if "_v" not in experiment_name:
            # Find the latest version
            latest_version = self.get_latest_version(experiment_name)
            if latest_version:
                versioned_name = f"{experiment_name}_v{latest_version}"
                logger.info(f"Using latest version: {versioned_name}")
                return registry.get(versioned_name)

        logger.warning(f"Experiment {experiment_name} not found in registry")
        return None

    def get_latest_version(self, base_experiment_name: str) -> Optional[int]:
        """
        Get the latest version number for an experiment.

        Args:
            base_experiment_name: Base name of the experiment without version

        Returns:
            Latest version number or None if no versions exist
        """
        registry = self._load_registry()
        versions = []

        for name in registry.keys():
            if name.startswith(f"{base_experiment_name}_v"):
                try:
                    version = int(name.split("_v")[1])
                    versions.append(version)
                except ValueError:
                    continue

        if versions:
            return max(versions)
        return None

    def list_experiments(
        self,
        filter_tag: Optional[str] = None,
        filter_status: Optional[str] = None,
        sort_by: str = "creation_time",
    ) -> List[Dict]:
        """
        List all registered experiments, optionally filtered and sorted.

        Args:
            filter_tag: Optional tag to filter experiments
            filter_status: Optional status to filter experiments
            sort_by: Field to sort by (creation_time, last_updated, name)

        Returns:
            List of experiment records
        """
        registry = self._load_registry()
        experiments = []

        for name, data in registry.items():
            # Add the name to the experiment data
            exp_data = data.copy()
            exp_data["name"] = name

            # Apply tag filter if specified
            if filter_tag and "tags" in data:
                if filter_tag not in data["tags"]:
                    continue

            # Apply status filter if specified
            if filter_status and "status" in data:
                if data["status"] != filter_status:
                    continue

            experiments.append(exp_data)

        # Sort experiments
        if sort_by == "name":
            experiments.sort(key=lambda x: x["name"])
        elif sort_by == "last_updated" and all(
            "last_updated" in exp for exp in experiments
        ):
            experiments.sort(key=lambda x: x["last_updated"], reverse=True)
        else:  # Default to creation_time
            experiments.sort(key=lambda x: x.get("creation_time", ""), reverse=True)

        return experiments

    def delete_experiment(
        self, experiment_name: str, delete_files: bool = False
    ) -> bool:
        """
        Delete an experiment from the registry.

        Args:
            experiment_name: Name of the experiment
            delete_files: Whether to also delete associated files

        Returns:
            True if successful, False otherwise
        """
        registry = self._load_registry()
        if experiment_name not in registry:
            logger.warning(
                f"Cannot delete: Experiment {experiment_name} not found in registry"
            )
            return False

        # Remove from registry
        experiment_data = registry.pop(experiment_name)
        self._save_registry(registry)

        # Delete files if requested
        if delete_files and "path" in experiment_data:
            experiment_path = Path(experiment_data["path"])
            if experiment_path.exists():
                try:
                    import shutil

                    shutil.rmtree(experiment_path)
                    logger.info(f"Deleted experiment directory: {experiment_path}")
                except Exception as e:
                    logger.error(f"Error deleting experiment files: {e}")
                    return False

        logger.info(f"Deleted experiment: {experiment_name}")
        return True

    def get_experiment_comparison(
        self, experiment_names: List[str], metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare metrics between multiple experiments.

        Args:
            experiment_names: List of experiment names to compare
            metrics: Optional list of specific metrics to compare

        Returns:
            DataFrame with experiments as rows and metrics as columns
        """
        registry = self._load_registry()
        comparison_data = []

        for name in experiment_names:
            if name not in registry:
                logger.warning(f"Experiment {name} not found, skipping in comparison")
                continue

            experiment = registry[name]
            row_data = {"experiment": name}

            # Extract metrics
            if "metrics" in experiment:
                exp_metrics = experiment["metrics"]

                # If specific metrics requested
                if metrics:
                    for metric in metrics:
                        if metric in exp_metrics:
                            row_data[metric] = exp_metrics[metric]
                # Otherwise, get all top-level metrics (excluding 'steps')
                else:
                    for metric, value in exp_metrics.items():
                        if metric != "steps":
                            row_data[metric] = value

                # Add best value from steps if available
                if "steps" in exp_metrics:
                    steps_data = exp_metrics["steps"]
                    best_values = {}

                    # Find best values across steps (assuming higher is better)
                    for step, step_metrics in steps_data.items():
                        for metric, value in step_metrics.items():
                            if isinstance(value, (int, float)):
                                if (
                                    metric not in best_values
                                    or value > best_values[metric]["value"]
                                ):
                                    best_values[metric] = {"value": value, "step": step}

                    # Add to row data with "best_" prefix
                    for metric, data in best_values.items():
                        row_data[f"best_{metric}"] = data["value"]
                        row_data[f"best_{metric}_step"] = data["step"]

            comparison_data.append(row_data)

        # Convert to DataFrame
        if comparison_data:
            return pd.DataFrame(comparison_data)
        else:
            return pd.DataFrame()

    def _load_registry(self) -> Dict:
        """
        Load the experiment registry from disk.

        Returns:
            Dictionary containing the experiment registry
        """
        if not self.registry_file.exists():
            return {}

        try:
            with open(self.registry_file) as f:
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
            with open(self.registry_file, "w") as f:
                json.dump(registry, f, indent=2, sort_keys=True)
        except Exception as e:
            logger.error(f"Error saving experiment registry: {e}")

    def _save_config(self, config_path: Path, config: Dict) -> None:
        """
        Save experiment configuration to disk.

        Args:
            config_path: Path where the config should be saved
            config: Configuration dictionary
        """
        try:
            # Determine format based on file extension
            if str(config_path).endswith(".json"):
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
            elif str(config_path).endswith(".yaml") or str(config_path).endswith(
                ".yml"
            ):
                import yaml

                with open(config_path, "w") as f:
                    yaml.dump(config, f)
            else:
                # Default to JSON if format not recognized
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")


# Create a singleton instance for easy import
experiment_registry = ExperimentRegistry()
