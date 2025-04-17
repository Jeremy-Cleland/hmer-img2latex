"""
Training and validation logic for the image-to-LaTeX model.
"""

import os
import random
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.utils import prepare_batch
from img2latex.training.metrics import compute_all_metrics, masked_accuracy
from img2latex.utils.logging import get_logger
from img2latex.utils.mps_utils import set_device
from img2latex.utils.path_utils import path_manager
from img2latex.utils.registry import experiment_registry

logger = get_logger(__name__, log_level="INFO")


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
        eval_cfg = config.get("evaluation", {})
        self.bleu_batches = eval_cfg.get("bleu_batches", 10)
        self.enhanced_samples = eval_cfg.get("enhanced_samples", 2)

        # Optimizer and loss
        self.learning_rate = training_config.get("learning_rate", 0.001)
        self.weight_decay = training_config.get("weight_decay", 0.0001)

        # Gradient accumulation steps
        self.accumulation_steps = training_config.get("accumulation_steps", 1)
        logger.info(f"Using gradient accumulation with {self.accumulation_steps} steps")

        # Create optimizer
        self.optimizer = optim.Adam(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        # Create learning rate scheduler (ReduceLROnPlateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )
        # Mixed precision training (AMP) on CUDA or MPS
        if self.device.type in ("cuda", "mps"):
            self.use_amp = True
            # Only CUDA supports GradScaler; MPS will use autocast without scaling
            if self.device.type == "cuda":
                self.scaler = torch.amp.GradScaler()
            else:
                self.scaler = None
        else:
            self.use_amp = False
            self.scaler = None

        # Create loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            reduction="mean",
            label_smoothing=0.1,
        )

        # Other training parameters
        self.max_epochs = training_config.get("epochs", 50)
        self.grad_clip_norm = training_config.get("clip_grad_norm", 5.0)
        self.early_stopping_patience = training_config.get(
            "early_stopping_patience", 10
        )

        # Use save_checkpoint_epochs if available, otherwise fall back to steps
        self.save_checkpoint_epochs = training_config.get("save_checkpoint_epochs", 10)
        self.save_checkpoint_steps = training_config.get("save_checkpoint_steps", 1000)
        self.use_epoch_checkpointing = "save_checkpoint_epochs" in training_config

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
            Dictionary of training metrics for the epoch
        """
        # Set model to training mode
        self.model.train()

        # Initialize epoch metrics
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_tokens = 0
        epoch_samples = 0

        # Get progress bar settings from config
        logging_config = self.config.get("logging", {})
        batch_log_frequency = logging_config.get("batch_log_frequency", 5)
        detailed_log_frequency = logging_config.get("detailed_log_frequency", 50)

        # Create progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs}",
            leave=True,
        )

        # Reset gradients
        self.optimizer.zero_grad(set_to_none=True)

        # Iterate over batches
        for batch_idx, batch in enumerate(pbar):
            images, formulas = prepare_batch(batch, self.device)
            # Determine targets (exclude start token)
            targets = formulas[:, 1:]
            # Select training path based on AMP and accumulation
            if self.accumulation_steps == 1:
                # Single-step update
                if self.use_amp:
                    # Mixed precision forward for CUDA/MPS
                    with torch.autocast(device_type=self.device.type):
                        outputs = self.model(images, formulas)
                        logits = outputs.transpose(1, 2)
                        loss = self.criterion(logits, targets)
                    # Backward and optimizer step
                    if self.scaler is not None:
                        # CUDA GradScaler path
                        self.scaler.scale(loss).backward()
                        if self.grad_clip_norm > 0:
                            self.scaler.unscale_(self.optimizer)
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # MPS/autocast without scaler
                        loss.backward()
                        if self.grad_clip_norm > 0:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                else:
                    # FP32 path
                    outputs = self.model(images, formulas)
                    logits = outputs.transpose(1, 2)
                    loss = self.criterion(logits, targets)
                    loss.backward()
                    if self.grad_clip_norm > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip_norm
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                # Gradient accumulation
                if self.use_amp:
                    # Mixed precision forward
                    with torch.autocast(device_type=self.device.type):
                        outputs = self.model(images, formulas)
                        logits = outputs.transpose(1, 2)
                        loss = self.criterion(logits, targets) / self.accumulation_steps
                    # Backward
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    # At accumulation step, update weights
                    if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx == len(self.train_loader) - 1:
                        if self.scaler is not None:
                            # CUDA GradScaler path
                            if self.grad_clip_norm > 0:
                                self.scaler.unscale_(self.optimizer)
                                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            # MPS / FP32 path
                            if self.grad_clip_norm > 0:
                                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                            self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                else:
                    # Standard FP32 accumulation
                    outputs = self.model(images, formulas)
                    logits = outputs.transpose(1, 2)
                    loss = self.criterion(logits, targets) / self.accumulation_steps
                    loss.backward()
                    # At accumulation step, update weights
                    if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx == len(self.train_loader) - 1:
                        if self.grad_clip_norm > 0:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)

            # Update epoch metrics
            # Undo the loss normalization for metric tracking
            if self.accumulation_steps > 1:
                loss.item() * self.accumulation_steps
            batch_size = formulas.size(0)
            batch_tokens = (targets != self.tokenizer.pad_token_id).sum().item()
            batch_acc, batch_tokens = masked_accuracy(
                outputs, targets, self.tokenizer.pad_token_id
            )

            # Update running metrics
            epoch_loss += loss.item() * batch_size
            epoch_correct += (
                batch_acc  # batch_acc is already the count of correct tokens
            )
            epoch_tokens += batch_tokens
            epoch_samples += batch_size

            # Increment global step
            self.global_step += 1

            # Save checkpoint based on steps
            if (
                not self.use_epoch_checkpointing
                and self.global_step % self.save_checkpoint_steps == 0
            ):
                metrics = {
                    "train_loss": epoch_loss / epoch_samples,
                    "train_acc": epoch_correct / epoch_tokens,
                    "epoch": self.current_epoch,
                    "step": self.global_step,
                }
                self.save_checkpoint(
                    self.current_epoch, self.global_step, metrics, is_best=False
                )

            # Update progress bar
            if batch_idx % batch_log_frequency == 0:
                pbar.set_description(f"Loss: {loss.item():.4f}")

            if batch_idx % detailed_log_frequency == 0:
                # More detailed logging at less frequent intervals
                logger.info(
                    f"Epoch {self.current_epoch + 1}/{self.max_epochs}, "
                    f"Batch {batch_idx + 1}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Acc: {batch_acc / batch_tokens if batch_tokens > 0 else 0:.4f}"
                )

        # Calculate epoch metrics
        epoch_metrics = {
            "train_loss": epoch_loss / epoch_samples,
            "train_acc": epoch_correct / epoch_tokens if epoch_tokens > 0 else 0,
            "train_samples": epoch_samples,
            "epoch": self.current_epoch,
            "step": self.global_step,
        }

        # Log epoch metrics
        logger.info(
            f"Epoch {self.current_epoch + 1}/{self.max_epochs} - "
            f"Train Loss: {epoch_metrics['train_loss']:.4f}, "
            f"Train Acc: {epoch_metrics['train_acc']:.4f}"
        )

        # Save epoch checkpoint if using epoch-based checkpointing
        if (
            self.use_epoch_checkpointing
            and (self.current_epoch + 1) % self.save_checkpoint_epochs == 0
        ):
            self.save_checkpoint(
                self.current_epoch, self.global_step, epoch_metrics, is_best=False
            )

        return epoch_metrics

    def validate(self) -> Dict[str, float]:
        """
        Validate the model on the validation set.

        Returns:
            Dictionary of validation metrics
        """
        # Set model to evaluation mode
        self.model.eval()

        # Get evaluation config
        eval_cfg = self.config.get("evaluation", {})

        # Get logging config
        logging_config = self.config.get("logging", {})
        val_log_frequency = logging_config.get("val_log_frequency", 25)
        detailed_eval_frequency = logging_config.get("detailed_eval_frequency", 5)

        # Initialize validation metrics
        val_loss = 0.0
        val_correct = 0.0
        val_tokens = 0
        val_samples = 0

        # Lists to store predictions and targets for BLEU calculation
        all_predictions = []
        all_targets = []

        # We'll sample a fraction of batches rather than just the first few
        # Calculate a sampling rate to get approximately self.bleu_batches worth of data
        total_batches = len(self.val_loader)
        if total_batches > self.bleu_batches:
            sampling_rate = self.bleu_batches / total_batches
        else:
            sampling_rate = 1.0  # Sample all batches if we have fewer than bleu_batches

        logger.debug(
            f"Validation sampling rate: {sampling_rate:.4f} (targeting ~{self.bleu_batches} batches)"
        )

        # Create progress bar
        pbar = tqdm(
            self.val_loader,
            desc=f"Validation (Epoch {self.current_epoch + 1})",
            leave=True,
        )

        with torch.no_grad():
            # Iterate over batches
            for batch_idx, batch in enumerate(pbar):
                # Extract data
                images, formulas = prepare_batch(
                    batch, self.device, model_type=self.config["model"]["name"]
                )

                # Forward pass
                outputs = self.model(images, formulas)

                # Calculate loss
                logits = outputs.transpose(1, 2)
                targets = formulas[:, 1:]  # Exclude START token
                loss = self.criterion(logits, targets)

                # Update metrics
                batch_size = formulas.size(0)
                batch_acc, batch_tokens = masked_accuracy(
                    outputs, targets, self.tokenizer.pad_token_id
                )

                val_loss += loss.item() * batch_size
                val_correct += batch_acc
                val_tokens += batch_tokens
                val_samples += batch_size

                # Store predictions and targets using a sampling strategy
                # Either sample the first few batches or randomly sample throughout validation
                if (batch_idx < self.bleu_batches) or (random.random() < sampling_rate):
                    # Get the predicted tokens
                    pred_tokens = torch.argmax(outputs, dim=-1).cpu().numpy()
                    true_tokens = targets.cpu().numpy()

                    # Loop through batch
                    for i in range(batch_size):
                        # Convert to list and remove padding
                        pred_list = pred_tokens[i].tolist()
                        true_list = true_tokens[i].tolist()

                        # Remove padding tokens
                        pred_clean = []
                        for token in pred_list:
                            if token == self.tokenizer.pad_token_id:
                                break
                            pred_clean.append(token)

                        true_clean = []
                        for token in true_list:
                            if token == self.tokenizer.pad_token_id:
                                break
                            true_clean.append(token)

                        # Store cleaned sequences
                        all_predictions.append(pred_clean)
                        all_targets.append(true_clean)

                # Update progress bar
                if batch_idx == 0:
                    pbar.set_description(
                        f"Val Loss: {loss.item():.4f}, Val Acc: {batch_acc / batch_tokens if batch_tokens > 0 else 0:.4f}"
                    )

                # Log intermediate results
                if batch_idx % val_log_frequency == 0:
                    logger.info(
                        f"Validation Batch {batch_idx + 1}/{len(self.val_loader)}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Acc: {batch_acc / batch_tokens if batch_tokens > 0 else 0:.4f}"
                    )

                if batch_idx % 25 == 0 and batch_idx > 0:
                    # Show a sample prediction vs target
                    if len(all_predictions) > 0:
                        # Use a random index instead of batch_idx to show different samples
                        idx = random.randint(0, len(all_predictions) - 1)
                        pred_text = self.tokenizer.decode(all_predictions[idx])
                        true_text = self.tokenizer.decode(all_targets[idx])
                        logger.info(f"Sample prediction: {pred_text}")
                        logger.info(f"Sample target: {true_text}")

        # Calculate overall validation metrics
        val_metrics = {
            "val_loss": val_loss / val_samples,
            "val_acc": val_correct / val_tokens if val_tokens > 0 else 0,
            "val_samples": val_samples,
            "epoch": self.current_epoch,
            "step": self.global_step,
        }

        # Debug logging for accuracy calculation
        logger.debug(
            f"Validation accuracy details: correct_tokens={val_correct}, "
            f"total_tokens={val_tokens}, ratio={val_metrics['val_acc']:.6f}"
        )

        # Compute additional metrics if predictions were collected
        if all_predictions and all_targets:
            # Log the number of samples collected for metrics
            logger.info(
                f"Collected {len(all_predictions)} samples for BLEU and other metrics"
            )

            # Calculate enhanced metrics (e.g., BLEU)
            use_detailed_metrics = (
                self.current_epoch % detailed_eval_frequency == 0
                or self.current_epoch == self.max_epochs - 1
            )

            bleu_metrics = compute_all_metrics(
                outputs=None,
                targets=None,
                all_predictions=all_predictions,
                all_targets=all_targets,
                tokenizer=self.tokenizer,
                num_samples=self.enhanced_samples,
                experiment_name=self.experiment_name if use_detailed_metrics else None,
                metrics_dir=path_manager.get_metrics_dir(self.experiment_name)
                if use_detailed_metrics
                else None,
                save_to_file=use_detailed_metrics,
                epoch=self.current_epoch,
            )

            # Merge metrics
            val_metrics.update(bleu_metrics)

        # Log validation metrics
        logger.info(
            f"Validation (Epoch {self.current_epoch + 1}) - "
            f"Loss: {val_metrics['val_loss']:.4f}, "
            f"Acc: {val_metrics['val_acc']:.4f}"
            f", BLEU: {val_metrics.get('val_bleu', 0):.4f}"
        )

        # Add Levenshtein distance if available
        if "val_levenshtein" in val_metrics:
            logger.info(
                f"Validation metrics - "
                f", Levenshtein: {val_metrics.get('val_levenshtein', 0):.4f}"
            )

        return val_metrics

    def train(self) -> Dict[str, float]:
        """
        Train the model for the specified number of epochs.

        Returns:
            Dictionary of best validation metrics
        """
        # Update experiment status
        experiment_registry.update_experiment_status(self.experiment_name, "training")

        # Set smaller batch sizes for MPS if needed
        if self.device.type == "mps":
            # Check available MPS memory
            rec_max = torch.mps.recommended_max_memory()
            # If we have less than 6GB of available memory, reduce batch sizes
            if rec_max < 6 * (1024**3):
                train_batch_size = self.config.get("data", {}).get("batch_size", 128)
                # Use fallback_batch_size from config if available, otherwise default to 32
                fallback_batch_size = self.config.get("data", {}).get(
                    "fallback_batch_size", 32
                )
                if train_batch_size > fallback_batch_size:
                    new_batch_size = fallback_batch_size
                    logger.warning(
                        f"Limited MPS memory detected ({rec_max / (1024**3):.2f}GB). Reducing batch size from {train_batch_size} to {new_batch_size}"
                    )
                    # Set environment variable for batch size override
                    os.environ["MPS_BATCH_SIZE_OVERRIDE"] = str(new_batch_size)

        # Training loop
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch

            # Clean up memory at the start of each epoch
            if self.device.type == "mps":
                from img2latex.utils.mps_utils import deep_clean_memory

                deep_clean_memory()

            try:
                # Train for one epoch
                train_metrics = self.train_epoch()

                # Clean up memory between training and validation
                if self.device.type == "mps":
                    from img2latex.utils.mps_utils import deep_clean_memory

                    deep_clean_memory()

                # Validate
                val_metrics = self.validate()
                # Step LR scheduler based on validation loss
                try:
                    self.scheduler.step(val_metrics.get("val_loss", float("inf")))
                except Exception:
                    logger.warning(
                        "LR scheduler step failed; skipping scheduler update."
                    )

                # Explicit cache clearing after validation
                if self.device.type == "mps":
                    torch.mps.empty_cache()

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

                    # Clear cache after non-improvement checkpoint
                    if self.device.type == "mps":
                        torch.mps.empty_cache()

                    # Early stopping
                    if self.patience_counter >= self.early_stopping_patience:
                        logger.info(
                            f"Early stopping after {epoch + 1} epochs "
                            f"({self.patience_counter} epochs without improvement)"
                        )
                        break

                    # Ensure cache is cleared at end of epoch loop
                    if self.device.type == "mps":
                        torch.mps.empty_cache()

                    # Clean up memory at the end of each epoch
                    if self.device.type == "mps":
                        from img2latex.utils.mps_utils import deep_clean_memory

                        deep_clean_memory()

                        # Add diagnostics for MPS memory usage
                        print(
                            f"Epoch {epoch + 1} - MPS allocated: {torch.mps.current_allocated_memory() / (1024**2):.2f} MB"
                        )

            except RuntimeError as e:
                # Handle out of memory errors gracefully
                if "out of memory" in str(e).lower():
                    logger.error(f"Out of memory error during epoch {epoch + 1}: {e}")

                    # Clean memory and try to reduce batch size for next epoch
                    if self.device.type == "mps":
                        from img2latex.utils.mps_utils import deep_clean_memory

                        deep_clean_memory()

                        # Try to reduce batch size
                        if hasattr(self.train_loader, "batch_sampler") and hasattr(
                            self.train_loader.batch_sampler, "batch_size"
                        ):
                            current_size = self.train_loader.batch_sampler.batch_size
                            new_size = max(8, current_size // 2)  # Don't go below 8

                            if new_size < current_size:
                                logger.warning(
                                    f"Reducing batch size from {current_size} to {new_size} due to OOM error"
                                )
                                self.train_loader.batch_sampler.batch_size = new_size

                                # Also adjust val loader if possible
                                if hasattr(self.val_loader, "batch_sampler"):
                                    self.val_loader.batch_sampler.batch_size = new_size

                                # Continue to next epoch with reduced batch size
                                continue

                    # If we can't adjust, re-raise the error
                    raise
                else:
                    # Re-raise other errors
                    raise

        # Update experiment status
        experiment_registry.update_experiment_status(self.experiment_name, "completed")

        return self.best_val_metrics
