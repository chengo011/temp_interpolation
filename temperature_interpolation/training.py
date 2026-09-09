from datetime import datetime, timezone
from pathlib import Path
import json
import random
import time
import numpy as np
import pandas as pd
import torch

from .gaps import RandomGapBatches, evaluation_batches
from .model import build_model, masked_mse

"""Training with fixed validation, early stopping, and best-checkpoint recovery."""

def set_reproducibility(seed, threads):
    """Seed all generators and require deterministic PyTorch operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@torch.inference_mode()
def validation_loss(model, values, manifest, config, device):
    """Aggregate squared errors by masked POINTS, not by batch or window."""
    model.eval()
    squared_error, count = 0.0, 0
    for (inputs, truth, mask), _ in evaluation_batches(values, manifest, config):
        predictions = model(inputs.to(device))
        selected = mask.to(device).bool()
        error = predictions[selected] - truth.to(device)[selected]
        squared_error += error.square().sum().item()
        count += error.numel()
    return squared_error / count


def train_model(train_values, validation_values, validation_manifest, config, output_dir: Path):
    """Fit without a test-data argument and reload the best validation state.

    The learning-rate scheduler and checkpoint selection observe only the
    fixed 2015 validation gaps. The final epoch is never assumed to be best.
    """
    set_reproducibility(config.seed, config.threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_model.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir}")
    config.save(output_dir / "config.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = train_values.shape[1] + len(config.sensor_features)
    model = build_model(config, input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.scheduler_patience,
    )
    sampler = RandomGapBatches(train_values, config)
    history, best_loss, waiting = [], float("inf"), 0
    started = datetime.now(timezone.utc).isoformat()
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        squared_error, count = 0.0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for inputs, truth, mask in sampler.epoch(epoch):
            inputs, truth, mask = inputs.to(device), truth.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs)
            loss = masked_mse(predictions, truth, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became non-finite.")
            loss.backward()
            # Gradient clipping prevents rare unstable recurrent updates.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            points = int(mask.sum().item())
            squared_error += loss.item() * points
            count += points
        current_validation = validation_loss(model, validation_values, validation_manifest, config, device)
        if not np.isfinite(current_validation):
            raise FloatingPointError("Validation loss became non-finite.")
        history.append({
            "epoch": epoch, "train_masked_mse": squared_error / count,
            "validation_masked_mse": current_validation, "learning_rate": learning_rate,
            "seconds": time.perf_counter() - epoch_start,
        })
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        if current_validation < best_loss:
            best_loss, waiting = current_validation, 0
            torch.save({
                "state_dict": model.state_dict(), "input_size": input_size,
                "best_epoch": epoch, "validation_masked_mse": best_loss,
            }, checkpoint)
        else:
            waiting += 1
        scheduler.step(current_validation)
        print(f"{config.name}: epoch {epoch:02d}, train={squared_error/count:.6f}, "
              f"val={current_validation:.6f}, {history[-1]['seconds']:.1f}s", flush=True)
        if waiting >= config.early_stopping_patience:
            break
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(config, input_size)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    metadata = {
        "started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "torch_version": torch.__version__,
        "best_epoch": saved["best_epoch"], "best_validation_masked_mse": best_loss,
        "epochs_completed": len(history), "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_candidates_per_length": {str(k): len(v) for k, v in sampler.candidates.items()},
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model


def load_trained_model(output_dir, config):
    """Load trusted local model weights for evaluation or interpolation."""
    saved = torch.load(output_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model = build_model(config, saved["input_size"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model
