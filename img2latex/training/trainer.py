"""
Training and validation logic for the image-to-LaTeX model.
"""

import json
import math
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from img2latex.data.tokenizer import LaTeXTokenizer
from img2latex.data.utils import prepare_batch
from img2latex.training.metrics import calculate_metrics, masked_accuracy
from img2latex.utils.logging import get_logger
from img2latex.utils.mps_utils import set_device
from img2latex.utils.path_utils import path_manager
from img2latex.utils.registry import experiment_registry

logger = get_logger(__name__, log_level="INFO")


def _build_warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Trainer for the image-to-LaTeX model."""

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
        if device is None:
            device_name = config.get("training", {}).get("device", None)
            self.device = set_device(device_name)
        else:
            self.device = device

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.experiment_name = experiment_name

        training_config = config.get("training", {})
        eval_cfg = config.get("evaluation", {})
        self.gen_samples = eval_cfg.get("gen_samples", 256)
        self.enhanced_samples = eval_cfg.get("enhanced_samples", 2)
        self.selection_metric = eval_cfg.get("selection_metric", "bleu")
        self.val_beam_size = eval_cfg.get("beam_size", 1)
        inf_cfg = config.get("inference", {})
        self.length_penalty = inf_cfg.get("length_penalty", 0.7)

        self.learning_rate = training_config.get("learning_rate", 3e-4)
        self.weight_decay = training_config.get("weight_decay", 1e-4)
        self.accumulation_steps = training_config.get("accumulation_steps", 1)
        self.use_amp = bool(training_config.get("use_amp", False))
        logger.info("Gradient accumulation steps: %s", self.accumulation_steps)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98),
        )
        self.max_epochs = training_config.get("epochs", 30)
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / self.accumulation_steps))
        total_steps = steps_per_epoch * self.max_epochs
        warmup_steps = int(training_config.get("warmup_ratio", 0.05) * total_steps)
        self.scheduler = _build_warmup_cosine(self.optimizer, warmup_steps, total_steps)
        logger.info("AdamW warmup %s / %s optimizer steps", warmup_steps, total_steps)

        self.scaler = None
        if self.use_amp and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler()
        elif self.use_amp:
            logger.warning("AMP requested but no GradScaler on %s; running fp32", self.device.type)
            self.use_amp = False

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id,
            reduction="mean",
            label_smoothing=0.1,
        )
        self.grad_clip_norm = training_config.get("clip_grad_norm", 1.0)
        self.early_stopping_patience = training_config.get("early_stopping_patience", 8)
        self.save_checkpoint_epochs = training_config.get("save_checkpoint_epochs", 10)
        self.save_checkpoint_steps = training_config.get("save_checkpoint_steps", 1000)
        self.use_epoch_checkpointing = "save_checkpoint_epochs" in training_config

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_bleu = -1.0
        self.best_val_loss = float("inf")
        self.best_val_metrics = {}
        self.patience_counter = 0
        self._register_experiment()
        logger.info(
            "Initialized trainer for '%s' on '%s' with %s epochs",
            experiment_name,
            self.device,
            self.max_epochs,
        )

    def _register_experiment(self) -> None:
        training_config = self.config.get("training", {})
        model_config = self.config.get("model", {})
        model_type = model_config.get("name", "cnn_transformer")
        description = (
            f"Image-to-LaTeX model: {model_type} d_model={model_config.get('embedding_dim')}. "
            f"lr={training_config.get('learning_rate')}, epochs={self.max_epochs}"
        )
        experiment_registry.register_experiment(
            experiment_name=self.experiment_name,
            config=self.config,
            description=description,
            tags=[model_type, f"lr_{training_config.get('learning_rate')}"],
        )
        experiment_registry.update_experiment_status(self.experiment_name, "initialized")

    def save_checkpoint(self, epoch: int, step: int, metrics: Dict[str, float], is_best: bool = False) -> str:
        checkpoint_dir = experiment_registry.path_manager.get_checkpoint_dir(self.experiment_name)
        checkpoint_name = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        if is_best:
            checkpoint_name = f"best_{checkpoint_name}"
        checkpoint_path = checkpoint_dir / checkpoint_name
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "tokenizer_config": {
                "token_to_id": self.tokenizer.token_to_id,
                "special_tokens": self.tokenizer.special_tokens,
                "max_sequence_length": self.tokenizer.max_sequence_length,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved checkpoint to %s", checkpoint_path)
        if is_best:
            best_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info("Saved best checkpoint to %s", best_path)
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["step"]
        metrics = checkpoint.get("metrics", {})
        if "val_bleu" in metrics:
            self.best_val_bleu = metrics["val_bleu"]
        if "val_loss" in metrics:
            self.best_val_loss = metrics["val_loss"]
        logger.info("Loaded checkpoint from %s (epoch %s, step %s)", checkpoint_path, self.current_epoch, self.global_step)
        return checkpoint

    def _forward_loss(self, images, formulas, widths, heights):
        outputs = self.model(images, formulas, valid_widths=widths, valid_heights=heights)
        targets = formulas[:, 1:]
        logits = outputs.transpose(1, 2)
        raw_loss = self.criterion(logits, targets)
        return outputs, targets, raw_loss

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_tokens = 0
        epoch_samples = 0
        logging_config = self.config.get("logging", {})
        batch_log_frequency = logging_config.get("batch_log_frequency", 5)
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.max_epochs}",
            leave=True,
        )
        self.optimizer.zero_grad(set_to_none=True)
        model_type = self.config.get("model", {}).get("name", "cnn_transformer")

        for batch_idx, batch in enumerate(pbar):
            images, formulas, widths, heights = prepare_batch(batch, self.device, model_type=model_type)
            outputs, targets, raw_loss = self._forward_loss(images, formulas, widths, heights)
            loss = raw_loss / self.accumulation_steps
            loss.backward()

            stepped = False
            if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx == len(self.train_loader) - 1:
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                stepped = True

            batch_size = formulas.size(0)
            batch_acc, batch_tokens = masked_accuracy(outputs, targets, self.tokenizer.pad_token_id)
            epoch_loss += raw_loss.item() * batch_size
            epoch_correct += batch_acc
            epoch_tokens += batch_tokens
            epoch_samples += batch_size
            if stepped:
                self.global_step += 1

            if batch_idx % batch_log_frequency == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                acc = batch_acc / batch_tokens if batch_tokens else 0.0
                pbar.set_description(f"Loss: {raw_loss.item():.4f} Acc: {acc:.3f} lr: {lr:.2e}")

        epoch_metrics = {
            "train_loss": epoch_loss / max(epoch_samples, 1),
            "train_acc": epoch_correct / epoch_tokens if epoch_tokens > 0 else 0,
            "train_samples": epoch_samples,
            "epoch": self.current_epoch,
            "step": self.global_step,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        logger.info(
            "Epoch %s/%s - Train Loss: %.4f, Train Acc: %.4f",
            self.current_epoch + 1,
            self.max_epochs,
            epoch_metrics["train_loss"],
            epoch_metrics["train_acc"],
        )
        if self.use_epoch_checkpointing and (self.current_epoch + 1) % self.save_checkpoint_epochs == 0:
            self.save_checkpoint(self.current_epoch, self.global_step, epoch_metrics, is_best=False)
        return epoch_metrics

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        model_type = self.config.get("model", {}).get("name", "cnn_transformer")
        val_loss = 0.0
        val_correct = 0.0
        val_tokens = 0
        val_samples = 0
        all_predictions = []
        all_targets = []
        remaining = self.gen_samples
        pbar = tqdm(self.val_loader, desc=f"Validation (Epoch {self.current_epoch + 1})", leave=True)

        with torch.no_grad():
            for batch in pbar:
                images, formulas, widths, heights = prepare_batch(batch, self.device, model_type=model_type)
                outputs, targets, raw_loss = self._forward_loss(images, formulas, widths, heights)
                batch_size = formulas.size(0)
                batch_acc, batch_tokens = masked_accuracy(outputs, targets, self.tokenizer.pad_token_id)
                val_loss += raw_loss.item() * batch_size
                val_correct += batch_acc
                val_tokens += batch_tokens
                val_samples += batch_size

                if remaining > 0:
                    take = min(remaining, batch_size)
                    generated = self.model.generate(
                        images[:take],
                        start_token_id=self.tokenizer.start_token_id,
                        end_token_id=self.tokenizer.end_token_id,
                        max_length=self.tokenizer.max_sequence_length,
                        beam_size=max(self.val_beam_size, 1),
                        length_penalty=self.length_penalty,
                        valid_widths=None if widths is None else widths[:take],
                        valid_heights=None if heights is None else heights[:take],
                    )
                    for i, pred in enumerate(generated):
                        true_ids = targets[i].detach().cpu().tolist()
                        true_clean = []
                        for token in true_ids:
                            if token == self.tokenizer.pad_token_id:
                                continue
                            if token == self.tokenizer.end_token_id:
                                break
                            if token == self.tokenizer.start_token_id:
                                continue
                            true_clean.append(token)
                        all_predictions.append(pred)
                        all_targets.append(true_clean)
                    remaining -= take

        val_metrics = {
            "val_loss": val_loss / max(val_samples, 1),
            "val_acc": val_correct / val_tokens if val_tokens > 0 else 0,
            "val_samples": val_samples,
            "epoch": self.current_epoch,
            "step": self.global_step,
        }
        if all_predictions:
            gen_metrics = calculate_metrics(
                all_predictions,
                all_targets,
                pad_token_id=self.tokenizer.pad_token_id,
                start_token_id=self.tokenizer.start_token_id,
                end_token_id=self.tokenizer.end_token_id,
            )
            val_metrics["val_bleu"] = gen_metrics["bleu"]
            val_metrics["val_levenshtein"] = gen_metrics["levenshtein"]
            val_metrics["val_exact_match"] = gen_metrics["exact_match"]
            val_metrics["val_edit_distance"] = gen_metrics["edit_distance"]
            val_metrics["val_gen_samples"] = gen_metrics["batch_size"]
            if all_predictions:
                logger.info("Sample prediction: %s", self.tokenizer.decode(all_predictions[0]))
                logger.info("Sample target:     %s", self.tokenizer.decode(all_targets[0]))

        logger.info(
            "Validation (Epoch %s) - Loss: %.4f, TF Acc: %.4f, BLEU: %.4f, EM: %.4f, Edit: %.4f",
            self.current_epoch + 1,
            val_metrics["val_loss"],
            val_metrics["val_acc"],
            val_metrics.get("val_bleu", 0.0),
            val_metrics.get("val_exact_match", 0.0),
            val_metrics.get("val_edit_distance", 1.0),
        )
        return val_metrics

    def _write_metrics(self, metrics: Dict[str, float]) -> None:
        if not self.config.get("evaluation", {}).get("save_basic_metrics", False):
            return
        metrics_dir = experiment_registry.path_manager.get_metrics_dir(self.experiment_name)
        metrics_path = os.path.join(str(metrics_dir), "metrics.json")
        try:
            with open(metrics_path, "r") as handle:
                basic_metrics = json.load(handle)
        except Exception:
            basic_metrics = {}
        serializable = {
            k: (float(v) if isinstance(v, (float, int)) else v)
            for k, v in metrics.items()
            if not isinstance(v, (dict, list))
        }
        basic_metrics[str(self.current_epoch + 1)] = serializable
        with open(metrics_path, "w") as handle:
            json.dump(basic_metrics, handle, indent=2)

    def train(self) -> Dict[str, float]:
        experiment_registry.update_experiment_status(self.experiment_name, "training")
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            if self.device.type == "mps":
                from img2latex.utils.mps_utils import deep_clean_memory

                deep_clean_memory()
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            combined = {**train_metrics, **val_metrics}
            self._write_metrics(combined)
            current_bleu = val_metrics.get("val_bleu", 0.0)
            improved = current_bleu > self.best_val_bleu
            if improved:
                self.best_val_bleu = current_bleu
                self.best_val_loss = val_metrics["val_loss"]
                self.best_val_metrics = combined
                self.patience_counter = 0
                self.save_checkpoint(epoch, self.global_step, combined, is_best=True)
            else:
                self.patience_counter += 1
                logger.info(
                    "No BLEU improvement for %s epochs (best val_bleu: %.4f)",
                    self.patience_counter,
                    self.best_val_bleu,
                )
                self.save_checkpoint(epoch, self.global_step, combined, is_best=False)
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info("Early stopping after %s epochs", epoch + 1)
                    break
            if self.device.type == "mps":
                torch.mps.empty_cache()

        experiment_registry.update_experiment_status(self.experiment_name, "completed")
        return self.best_val_metrics
