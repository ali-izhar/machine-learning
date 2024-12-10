"""Training functionality for KAN Anomaly Detection"""

import torch
import numpy as np
from .utils import evaluate_metrics, find_optimal_threshold


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_val_f1 = 0
        self.optimal_threshold = 0.5

    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_preds_pos = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            out = self.model(x_batch)
            loss = self.criterion(out, y_batch)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            total_loss += loss.item() * x_batch.size(0)
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).float()
            acc = (preds == y_batch).float().mean().item()
            total_acc += acc * x_batch.size(0)
            total_preds_pos += preds.sum().item()

        return {
            "loss": total_loss / len(train_loader.dataset),
            "acc": total_acc / len(train_loader.dataset),
            "pos_preds": total_preds_pos,
        }

    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_loss = 0
        all_true = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                out = self.model(x_batch)
                loss = self.criterion(out, y_batch)
                val_loss += loss.item() * x_batch.size(0)

                probs = torch.sigmoid(out)
                all_true.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Find optimal threshold
        threshold, f1 = find_optimal_threshold(all_probs, all_true)
        all_preds = (np.array(all_probs) > threshold).astype(int)

        # Calculate metrics
        metrics = evaluate_metrics(all_true, all_preds, all_probs)

        return {
            "loss": val_loss / len(val_loader.dataset),
            "threshold": threshold,
            "f1": f1,
            "precision": metrics[0],
            "recall": metrics[1],
            "roc_auc": metrics[3],
        }

    def train(self, train_loader, val_loader, epochs, patience):
        """Full training loop with early stopping."""
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            train_metrics = self.train_epoch(train_loader)

            # Validation phase
            val_metrics = self.validate(val_loader)

            # Scheduler step
            self.scheduler.step()

            # Early stopping
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.optimal_threshold = val_metrics["threshold"]
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), "best_kan_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

            # Print metrics
            print(
                f"Epoch: {epoch+1:04d}, "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}, "
                f"Val ROC AUC: {val_metrics['roc_auc']:.4f}"
            )

        # Load best model
        self.model.load_state_dict(torch.load("best_kan_model.pth"))
        return self.optimal_threshold
