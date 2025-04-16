"""
Training and validation logic for the image-to-LaTeX model.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.utils import prepare_batch
from img2latex.training.enhanced_metrics import generate_enhanced_metrics
from img2latex.training.metrics import calculate_metrics, masked_accuracy
from img2latex.utils.logging import get_logger
from img2latex.utils.mps_utils import empty_cache, set_device
from img2latex.utils.registry import experiment_registry

logger = get_logger(__name__)


class Trainer:
    """
    Trainer for the image-to-LaTeX model.

    This class handles the training and validation of the model,
    as well as checkpointing and metric logging.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: LaTeXTokenizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        experiment_name: str,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Model to train
            tokenizer: Tokenizer for LaTeX formulas
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Configuration dictionary
            experiment_name: Name of the experiment
            device: Device to use for training (will be auto-detected if None)
        """
        # Set device
        if device is None:
            device_name = config.get("training", {}).get("device", None)
            self.device = set_device(device_name)
        else:
            self.device = device

        # Model and tokenizer
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training configuration
        self.config = config
        self.experiment_name = experiment_name

        # Get training parameters
        training_config = config.get("training", {})

        # Optimizer and loss
        self.learning_rate = training_config.get("learning_rate", 0.001)
        self.weight_decay = training_config.get("weight_decay", 0.0001)

        # Create optimizer
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Create loss function
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id, reduction="mean"
        )

        # Other training parameters
        self.max_epochs = training_config.get("epochs", 50)
        self.grad_clip_norm = training_config.get("clip_grad_norm", 5.0)
        self.early_stopping_patience = training_config.get(
            "early_stopping_patience", 10
        )
        self.save_checkpoint_steps = training_config.get("save_checkpoint_steps", 1000)

        # Initialize step and epoch counters
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_metrics = {}
        self.patience_counter = 0

        # Register the experiment
        self._register_experiment()

        logger.info(
            f"Initialized trainer for experiment '{experiment_name}' "
            f"on device '{self.device}' with {self.max_epochs} epochs"
        )

    def _register_experiment(self) -> None:
        """Register the experiment with the experiment registry."""
        # Extract relevant config for logging
        training_config = self.config.get("training", {})
        model_config = self.config.get("model", {})

        # Build experiment description
        model_type = model_config.get("name", "cnn_lstm")
        model_desc = (
            f"{model_type} with {model_config['embedding_dim']} embedding dim, "
            f"{model_config['decoder']['hidden_dim']} hidden dim"
        )

        # Create description
        description = (
            f"Image-to-LaTeX model: {model_desc}. "
            f"Training with lr={training_config['learning_rate']}, "
            f"batch_size={training_config.get('batch_size', 128)}, "
            f"max_epochs={self.max_epochs}"
        )

        # Create tags
        tags = [model_type, f"lr_{training_config['learning_rate']}"]

        # Register with experiment registry
        experiment_registry.register_experiment(
            experiment_name=self.experiment_name,
            config=self.config,
            description=description,
            tags=tags,
        )

        # Update experiment status
        experiment_registry.update_experiment_status(
            self.experiment_name, "initialized"
        )

    def save_checkpoint(
        self, epoch: int, step: int, metrics: Dict[str, float], is_best: bool = False
    ) -> str:
        """
        Save a checkpoint of the model.

        Args:
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to the saved checkpoint
        """
        # Get checkpoint directory
        checkpoint_dir = experiment_registry.path_manager.get_checkpoint_dir(
            self.experiment_name
        )

        # Create checkpoint name
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        if is_best:
            checkpoint_name = f"best_{checkpoint_name}"

        checkpoint_path = checkpoint_dir / checkpoint_name

        # Create a checkpoint dictionary
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "tokenizer_config": {
                "token_to_id": self.tokenizer.token_to_id,
                "special_tokens": self.tokenizer.special_tokens,
                "max_sequence_length": self.tokenizer.max_sequence_length,
            },
        }

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Also save best checkpoint separately
        if is_best:
            best_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint

        Returns:
            Loaded checkpoint dictionary
        """
        # Check if the checkpoint exists
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model and optimizer state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Update counters
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["step"]

        # Update best validation loss if available
        if "metrics" in checkpoint and "val_loss" in checkpoint["metrics"]:
            self.best_val_loss = checkpoint["metrics"]["val_loss"]

        logger.info(
            f"Loaded checkpoint from {checkpoint_path} "
            f"(epoch {self.current_epoch}, step {self.global_step})"
        )

        return checkpoint

    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_tokens = 0
        epoch_samples = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training epoch {self.current_epoch + 1}/{self.max_epochs}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch
            images, targets = prepare_batch(
                batch, self.device, model_type=self.config["model"]["name"]
            )

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, targets)

            # Reshape for loss calculation
            batch_size, seq_length, vocab_size = outputs.shape
            targets_shifted = targets[:, 1:]  # Skip the first token (START token)

            # Calculate loss
            loss = self.criterion(
                outputs.reshape(-1, vocab_size), targets_shifted.reshape(-1)
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip_norm
                )

            # Optimizer step
            self.optimizer.step()

            # Calculate accuracy
            acc, num_tokens = masked_accuracy(
                outputs, targets_shifted, self.tokenizer.pad_token_id
            )

            # Update counters
            epoch_loss += loss.item() * batch_size
            epoch_acc += acc * num_tokens
            epoch_tokens += num_tokens
            epoch_samples += batch_size
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item(), "acc": acc})

            # Save checkpoint if needed
            if self.global_step % self.save_checkpoint_steps == 0:
                metrics = {
                    "train_loss": epoch_loss / epoch_samples,
                    "train_acc": epoch_acc / epoch_tokens,
                }
                self.save_checkpoint(
                    epoch=self.current_epoch, step=self.global_step, metrics=metrics
                )

            # Clean up GPU memory in MPS mode
            if self.device.type == "mps" and batch_idx % 10 == 0:
                empty_cache()

        # Calculate epoch metrics
        metrics = {
            "train_loss": epoch_loss / epoch_samples,
            "train_acc": epoch_acc / epoch_tokens,
            "epoch": self.current_epoch + 1,
            "step": self.global_step,
        }

        # Log metrics
        experiment_registry.log_metrics(
            self.experiment_name, metrics, step=self.current_epoch + 1
        )

        logger.info(
            f"Epoch {self.current_epoch + 1}/{self.max_epochs}, "
            f"Train Loss: {metrics['train_loss']:.4f}, "
            f"Accuracy: {metrics['train_acc']:.4f}"
        )

        return metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model on the validation set.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        val_loss = 0.0
        val_acc = 0.0
        val_tokens = 0
        val_samples = 0

        # Lists to store predictions and targets for metric calculation
        all_predictions = []
        all_targets = []

        # Store outputs and targets for enhanced metrics
        enhanced_metrics_batch = None
        enhanced_metrics_targets = None

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Validation epoch {self.current_epoch + 1}/{self.max_epochs}",
            leave=False,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Prepare batch
                images, targets = prepare_batch(
                    batch, self.device, model_type=self.config["model"]["name"]
                )

                # Forward pass
                outputs = self.model(images, targets)

                # Reshape for loss calculation
                batch_size, seq_length, vocab_size = outputs.shape
                targets_shifted = targets[:, 1:]  # Skip the first token (START token)

                # Calculate loss
                loss = self.criterion(
                    outputs.reshape(-1, vocab_size), targets_shifted.reshape(-1)
                )

                # Calculate accuracy
                acc, num_tokens = masked_accuracy(
                    outputs, targets_shifted, self.tokenizer.pad_token_id
                )

                # Update counters
                val_loss += loss.item() * batch_size
                val_acc += acc * num_tokens
                val_tokens += num_tokens
                val_samples += batch_size

                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item(), "acc": acc})

                # Get predictions for metric calculation
                # We'll use a subset of the validation set for BLEU and Levenshtein metrics
                if batch_idx < 10:  # Limit to first 10 batches to save time
                    pred_ids = torch.argmax(outputs, dim=-1).cpu().numpy()
                    target_ids = targets_shifted.cpu().numpy()

                    # Convert to list of lists
                    for i in range(batch_size):
                        # Get masks for non-padding tokens
                        pred_mask = pred_ids[i] != self.tokenizer.pad_token_id
                        target_mask = target_ids[i] != self.tokenizer.pad_token_id

                        # Get non-padding tokens
                        pred_tokens = pred_ids[i][pred_mask].tolist()
                        target_tokens = target_ids[i][target_mask].tolist()

                        # Add to lists
                        all_predictions.append(pred_tokens)
                        all_targets.append(target_tokens)

                    # Store first batch for enhanced metrics
                    if batch_idx == 0:
                        enhanced_metrics_batch = outputs
                        enhanced_metrics_targets = targets_shifted

                # Clean up GPU memory in MPS mode
                if self.device.type == "mps" and batch_idx % 10 == 0:
                    empty_cache()

        # Calculate validation metrics
        metrics = {
            "val_loss": val_loss / val_samples,
            "val_acc": val_acc / val_tokens,
            "epoch": self.current_epoch + 1,
            "step": self.global_step,
        }

        # Calculate BLEU and Levenshtein metrics
        if all_predictions:
            extra_metrics = calculate_metrics(all_predictions, all_targets)
            metrics.update(
                {
                    "val_bleu": extra_metrics["bleu"],
                    "val_levenshtein": extra_metrics["levenshtein"],
                }
            )

            # Generate enhanced metrics if we have stored a batch
            if (
                enhanced_metrics_batch is not None
                and enhanced_metrics_targets is not None
            ):
                # Get experiment paths from registry
                experiment_paths = experiment_registry.get_experiment_paths(
                    self.experiment_name
                )
                metrics_dir = experiment_paths.get("metrics_dir", "")

                # Generate enhanced metrics
                generate_enhanced_metrics(
                    enhanced_metrics_batch,
                    enhanced_metrics_targets,
                    all_predictions,
                    all_targets,
                    self.tokenizer,
                    num_samples=5,
                    experiment_name=self.experiment_name,
                    metrics_dir=metrics_dir,
                    epoch=self.current_epoch,
                    save_to_file=True,
                )

        # Log metrics
        experiment_registry.log_metrics(
            self.experiment_name, metrics, step=self.current_epoch + 1
        )

        logger.info(
            f"Epoch {self.current_epoch + 1}/{self.max_epochs}, "
            f"Validation Loss: {metrics['val_loss']:.4f}, "
            f"Accuracy: {metrics['val_acc']:.4f}"
            + (
                f", BLEU: {metrics.get('val_bleu', 0):.4f}"
                if "val_bleu" in metrics
                else ""
            )
            + (
                f", Levenshtein: {metrics.get('val_levenshtein', 0):.4f}"
                if "val_levenshtein" in metrics
                else ""
            )
        )

        return metrics

    def train(self) -> Dict[str, float]:
        """
        Train the model for the specified number of epochs.

        Returns:
            Dictionary of best validation metrics
        """
        # Update experiment status
        experiment_registry.update_experiment_status(self.experiment_name, "training")

        # Training loop
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Check for improvement
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.best_val_metrics = val_metrics
                self.patience_counter = 0

                # Save best checkpoint
                self.save_checkpoint(
                    epoch=epoch,
                    step=self.global_step,
                    metrics=val_metrics,
                    is_best=True,
                )
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement for {self.patience_counter} epochs "
                    f"(best val_loss: {self.best_val_loss:.4f})"
                )

                # Save checkpoint
                self.save_checkpoint(
                    epoch=epoch, step=self.global_step, metrics=val_metrics
                )

                # Early stopping
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping after {epoch + 1} epochs "
                        f"({self.patience_counter} epochs without improvement)"
                    )
                    break

        # Update experiment status
        experiment_registry.update_experiment_status(self.experiment_name, "completed")

        return self.best_val_metrics
