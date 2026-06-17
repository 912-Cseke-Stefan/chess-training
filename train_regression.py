import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import data_preparation
import models
import scalers


def read_from_file(filename, output_parser):
    X = []
    y = []
    skipped = 0

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line[0] in {"F", "f"}:
                continue

            inputs, outputs = output_parser(line)
            if inputs is None:
                skipped += 1
                continue

            X.append(inputs)
            y.append(outputs)

    if not X:
        raise ValueError(f"No usable rows were found in {filename}.")

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X, y, skipped


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def split_regression_data(X, y, val_size, test_size, seed):
    if val_size <= 0 or test_size <= 0:
        raise ValueError("--val-size and --test-size must both be greater than 0.")
    if val_size + test_size >= 1:
        raise ValueError("--val-size + --test-size must be less than 1.")

    holdout_size = val_size + test_size
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=seed,
    )

    test_fraction_of_holdout = test_size / holdout_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_fraction_of_holdout,
        random_state=seed + 1,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def configure_model_for_stability(model, hidden_dropout, input_dropout, bn_momentum):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = input_dropout if name == "dropout" else hidden_dropout
        elif isinstance(module, torch.nn.BatchNorm1d):
            module.momentum = bn_momentum


def make_loader(X, y, batch_size, shuffle, device, num_workers, generator=None):
    return DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def clone_model_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio):
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    device,
    amp_scaler,
    use_amp,
    grad_clip,
):
    model.train()
    total_loss = 0.0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

        amp_scaler.scale(loss).backward()

        if grad_clip > 0:
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        amp_scaler.step(optimizer)
        amp_scaler.update()
        scheduler.step()

        batch_items = y_batch.numel()
        total_loss += loss.detach().item() * batch_items
        total += batch_items

    return total_loss / total


@torch.no_grad()
def evaluate_regression_model(model, loader, target_scaler, criterion, device):
    model.eval()

    scaled_loss_total = 0.0
    squared_error_total = 0.0
    absolute_error_total = 0.0
    total = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        scaled_y_pred = model(X_batch)
        scaled_y_batch = target_scaler.scale(y_batch)
        y_pred = target_scaler.reverse(scaled_y_pred.float())
        error = y_pred - y_batch

        batch_items = y_batch.numel()
        scaled_loss_total += criterion(scaled_y_pred, scaled_y_batch).item() * batch_items
        squared_error_total += torch.sum(error ** 2).item()
        absolute_error_total += torch.sum(torch.abs(error)).item()
        total += batch_items

    scaled_loss = scaled_loss_total / total
    rmse = (squared_error_total / total) ** 0.5
    mae = absolute_error_total / total
    return scaled_loss, rmse, mae


def get_artifact_paths(save_dir, model_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = Path(model_filename)
    stem = model_filename.stem
    suffix = model_filename.suffix or ".pth"

    model_path = save_dir / f"{stem}_{timestamp}{suffix}"
    progress_path = save_dir / f"progres_{stem}_{timestamp}.txt"
    metadata_path = save_dir / f"{stem}_{timestamp}.json"
    return model_path, progress_path, metadata_path


def json_safe_args(args):
    safe = {}
    for key, value in vars(args).items():
        safe[key] = str(value) if isinstance(value, Path) else value
    return safe


def save_progress(progress_path, rows):
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    with open(progress_path, "w", encoding="utf-8") as file:
        file.write("epoch,lr,train_scaled_loss,val_scaled_loss,val_rmse,val_mae,best_val_mae\n")
        for row in rows:
            file.write(
                f"{row['epoch']},"
                f"{row['lr']:.10g},"
                f"{row['train_scaled_loss']:.10g},"
                f"{row['val_scaled_loss']:.10g},"
                f"{row['val_rmse']:.10g},"
                f"{row['val_mae']:.10g},"
                f"{row['best_val_mae']:.10g}\n"
            )


def save_artifacts(
    model_state,
    progress_rows,
    metadata,
    save_dir,
    model_filename,
):
    model_path, progress_path, metadata_path = get_artifact_paths(save_dir, model_filename)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model_state, model_path)
    save_progress(progress_path, progress_rows)

    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return model_path, progress_path, metadata_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a more stable chess evaluation regression model."
    )
    parser.add_argument("--data", type=Path, default=Path("selected_top_level_games.csv"))
    parser.add_argument("--save-dir", type=Path, default=Path("save"))
    parser.add_argument("--model-filename", default="regression_stable.pth")
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--smooth-l1-beta", type=float, default=0.25)
    parser.add_argument("--target-scale", type=float, default=1000.0)
    parser.add_argument("--hidden-dropout", type=float, default=0.10)
    parser.add_argument("--input-dropout", type=float, default=0.0)
    parser.add_argument("--bn-momentum", type=float, default=0.03)

    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--print-every", type=int, default=5)

    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")

    if device_arg == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    seed_everything(args.seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    print(f"Using device: {device}")
    print(f"Loading data from {args.data}...")
    X, y, skipped = read_from_file(
        args.data,
        data_preparation.turn_fen_to_board_inputs_raw_output,
    )
    print(f"Loaded {len(X)} positions; skipped {skipped} unusable rows.")

    X_train, X_val, X_test, y_train, y_val, y_test = split_regression_data(
        X,
        y,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(
        f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
    )

    target_scaler = scalers.AsinhScaler(args.target_scale)
    scaled_y_train = target_scaler.scale(y_train)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = make_loader(
        X_train,
        scaled_y_train,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
        num_workers=args.num_workers,
        generator=loader_generator,
    )
    val_loader = make_loader(
        X_val,
        y_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        device=device,
        num_workers=args.num_workers,
    )
    test_loader = make_loader(
        X_test,
        y_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        device=device,
        num_workers=args.num_workers,
    )

    model = models.BoardInputRegression(X_train.shape[1]).to(device)
    configure_model_for_stability(
        model,
        hidden_dropout=args.hidden_dropout,
        input_dropout=args.input_dropout,
        bn_momentum=args.bn_momentum,
    )

    criterion = torch.nn.SmoothL1Loss(beta=args.smooth_l1_beta)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = args.warmup_epochs * len(train_loader)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    progress_rows = []

    initial_val_loss, initial_val_rmse, initial_val_mae = evaluate_regression_model(
        model,
        val_loader,
        target_scaler,
        criterion,
        device,
    )
    best_val_mae = initial_val_mae
    best_epoch = 0
    best_model_state = clone_model_state(model)
    epochs_since_best = 0

    progress_rows.append(
        {
            "epoch": 0,
            "lr": get_lr(optimizer),
            "train_scaled_loss": float("nan"),
            "val_scaled_loss": initial_val_loss,
            "val_rmse": initial_val_rmse,
            "val_mae": initial_val_mae,
            "best_val_mae": initial_val_mae,
        }
    )
    print(f"Epoch [0/{args.epochs}], Val MAE: {initial_val_mae:.2f} cp")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            amp_scaler=amp_scaler,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
        )

        val_loss, val_rmse, val_mae = evaluate_regression_model(
            model,
            val_loader,
            target_scaler,
            criterion,
            device,
        )

        improved = val_mae < best_val_mae - args.min_delta
        if improved:
            best_val_mae = val_mae
            best_epoch = epoch
            best_model_state = clone_model_state(model)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        progress_rows.append(
            {
                "epoch": epoch,
                "lr": get_lr(optimizer),
                "train_scaled_loss": train_loss,
                "val_scaled_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "best_val_mae": best_val_mae,
            }
        )

        should_print = epoch == 1 or epoch % args.print_every == 0 or improved
        if should_print:
            marker = " *" if improved else ""
            print(
                f"Epoch [{epoch}/{args.epochs}], "
                f"LR: {get_lr(optimizer):.2e}, "
                f"Train loss: {train_loss:.4f}, "
                f"Val loss: {val_loss:.4f}, "
                f"Val RMSE: {val_rmse:.2f} cp, "
                f"Val MAE: {val_mae:.2f} cp, "
                f"Best: {best_val_mae:.2f} cp @ {best_epoch}{marker}"
            )

        if epoch >= args.min_epochs and epochs_since_best >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best validation MAE was {best_val_mae:.2f} cp at epoch {best_epoch}."
            )
            break

    if best_model_state is None:
        best_model_state = clone_model_state(model)
        best_epoch = args.epochs

    model.load_state_dict(best_model_state)
    test_loss, test_rmse, test_mae = evaluate_regression_model(
        model,
        test_loader,
        target_scaler,
        criterion,
        device,
    )
    print(
        f"Best epoch: {best_epoch}; "
        f"test loss: {test_loss:.4f}, test RMSE: {test_rmse:.2f} cp, "
        f"test MAE: {test_mae:.2f} cp"
    )

    metadata = {
        "args": json_safe_args(args),
        "device": str(device),
        "amp": use_amp,
        "input_dim": X_train.shape[1],
        "model_class": "models.BoardInputRegression",
        "target_scaler": {
            "class": "scalers.AsinhScaler",
            "constant": args.target_scale,
        },
        "best_epoch": best_epoch,
        "best_val_mae": best_val_mae,
        "test_scaled_loss": test_loss,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
    }

    model_path, progress_path, metadata_path = save_artifacts(
        model_state=best_model_state,
        progress_rows=progress_rows,
        metadata=metadata,
        save_dir=args.save_dir,
        model_filename=args.model_filename,
    )
    print("-------------------------------")
    print(f"Saved best model to {model_path}")
    print(f"Saved progress to {progress_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
