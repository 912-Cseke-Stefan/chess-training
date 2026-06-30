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


def make_eval_bucket_labels(y):
    boundaries = torch.tensor(
        [-9000, -2000, -1000, -500, -300, -150, -50, 50, 150, 300, 500, 1000, 2000, 9000],
        dtype=torch.float32,
    )
    return torch.bucketize(y.detach().cpu().view(-1), boundaries).tolist()


def split_regression_data(X, y, val_size, test_size, seed, stratified=True):
    if val_size <= 0 or test_size <= 0:
        raise ValueError("--val-size and --test-size must both be greater than 0.")
    if val_size + test_size >= 1:
        raise ValueError("--val-size + --test-size must be less than 1.")

    holdout_size = val_size + test_size
    test_fraction_of_holdout = test_size / holdout_size
    indices = list(range(len(y)))

    if stratified:
        try:
            labels = make_eval_bucket_labels(y)
            train_indices, holdout_indices = train_test_split(
                indices,
                test_size=holdout_size,
                random_state=seed,
                stratify=labels,
            )
            holdout_labels = [labels[index] for index in holdout_indices]
            val_indices, test_indices = train_test_split(
                holdout_indices,
                test_size=test_fraction_of_holdout,
                random_state=seed + 1,
                stratify=holdout_labels,
            )
            split_method = "stratified_eval_bucket"
        except ValueError as error:
            print(f"Could not use stratified split ({error}); falling back to random split.")
            stratified = False

    if not stratified:
        train_indices, holdout_indices = train_test_split(
            indices,
            test_size=holdout_size,
            random_state=seed,
        )
        val_indices, test_indices = train_test_split(
            holdout_indices,
            test_size=test_fraction_of_holdout,
            random_state=seed + 1,
        )
        split_method = "random"

    return (
        X[train_indices],
        X[val_indices],
        X[test_indices],
        y[train_indices],
        y[val_indices],
        y[test_indices],
        split_method,
    )


def configure_model_for_stability(model, hidden_dropout, input_dropout, bn_momentum):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = input_dropout if name == "dropout" else hidden_dropout
        elif isinstance(module, torch.nn.BatchNorm1d):
            module.momentum = bn_momentum


def make_loader(*tensors, batch_size, shuffle, device, num_workers, generator=None):
    return DataLoader(
        TensorDataset(*tensors),
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


def update_ema_model(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)

        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            if torch.is_floating_point(ema_buffer):
                ema_buffer.mul_(decay).add_(buffer.detach(), alpha=1.0 - decay)
            else:
                ema_buffer.copy_(buffer)


def reverse_and_clip(target_scaler, scaled_y, prediction_clip):
    y = target_scaler.reverse(scaled_y.float())
    if prediction_clip > 0:
        y = torch.clamp(y, min=-prediction_clip, max=prediction_clip)
    return y


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
    target_scaler,
    optimizer,
    scheduler,
    device,
    amp_scaler,
    use_amp,
    grad_clip,
    loss_mode,
    raw_mae_weight,
    raw_mae_scale,
    prediction_clip,
    ema_model=None,
    ema_decay=0.0,
):
    model.train()
    total_loss_value = 0.0
    total_scaled_loss = 0.0
    total_raw_absolute_error = 0.0
    total = 0

    for X_batch, scaled_y_batch, raw_y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        scaled_y_batch = scaled_y_batch.to(device, non_blocking=True)
        raw_y_batch = raw_y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            scaled_y_pred = model(X_batch)
            scaled_loss = criterion(scaled_y_pred, scaled_y_batch)

        if loss_mode == "hybrid" and raw_mae_weight > 0:
            raw_y_pred = reverse_and_clip(target_scaler, scaled_y_pred, prediction_clip)
            raw_mae_loss = torch.mean(torch.abs(raw_y_pred - raw_y_batch.float()))
            loss = scaled_loss + raw_mae_weight * (raw_mae_loss / raw_mae_scale)
        else:
            loss = scaled_loss
            with torch.no_grad():
                raw_y_pred = reverse_and_clip(target_scaler, scaled_y_pred, prediction_clip)
                raw_mae_loss = torch.mean(torch.abs(raw_y_pred - raw_y_batch.float()))

        amp_scaler.scale(loss).backward()

        if grad_clip > 0:
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        amp_scaler.step(optimizer)
        amp_scaler.update()
        scheduler.step()

        if ema_model is not None and ema_decay > 0:
            update_ema_model(ema_model, model, ema_decay)

        batch_items = raw_y_batch.numel()
        total_loss_value += loss.detach().item() * batch_items
        total_scaled_loss += scaled_loss.detach().item() * batch_items
        total_raw_absolute_error += raw_mae_loss.detach().item() * batch_items
        total += batch_items

    return {
        "train_loss": total_loss_value / total,
        "train_scaled_loss": total_scaled_loss / total,
        "train_raw_mae": total_raw_absolute_error / total,
    }


@torch.no_grad()
def evaluate_regression_model(model, loader, target_scaler, criterion, device, prediction_clip):
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
        y_pred = reverse_and_clip(target_scaler, scaled_y_pred, prediction_clip)
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
        file.write(
            "epoch,lr,train_loss,train_scaled_loss,train_raw_mae,"
            "val_scaled_loss,val_rmse,val_mae,"
            "ema_val_scaled_loss,ema_val_rmse,ema_val_mae,"
            "selected_val_mae,best_val_mae,best_source\n"
        )
        for row in rows:
            file.write(
                f"{row['epoch']},"
                f"{row['lr']:.10g},"
                f"{row['train_loss']:.10g},"
                f"{row['train_scaled_loss']:.10g},"
                f"{row['train_raw_mae']:.10g},"
                f"{row['val_scaled_loss']:.10g},"
                f"{row['val_rmse']:.10g},"
                f"{row['val_mae']:.10g},"
                f"{row['ema_val_scaled_loss']:.10g},"
                f"{row['ema_val_rmse']:.10g},"
                f"{row['ema_val_mae']:.10g},"
                f"{row['selected_val_mae']:.10g},"
                f"{row['best_val_mae']:.10g},"
                f"{row['best_source']}\n"
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

    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--no-stratified-split", action="store_true")

    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.005)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--loss", choices=("scaled", "hybrid"), default="hybrid")
    parser.add_argument("--smooth-l1-beta", type=float, default=0.05)
    parser.add_argument("--raw-mae-weight", type=float, default=0.05)
    parser.add_argument("--raw-mae-scale", type=float, default=1000.0)
    parser.add_argument("--prediction-clip", type=float, default=10000.0)
    parser.add_argument("--target-scale", type=float, default=1000.0)
    parser.add_argument("--hidden-dropout", type=float, default=0.15)
    parser.add_argument("--input-dropout", type=float, default=0.0)
    parser.add_argument("--bn-momentum", type=float, default=0.03)

    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=120)
    parser.add_argument("--min-delta", type=float, default=0.05)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--ema-decay", type=float, default=0.999)

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
    if not 0 <= args.ema_decay < 1:
        raise ValueError("--ema-decay must be in the interval [0, 1). Use 0 to disable EMA.")
    if args.raw_mae_scale <= 0:
        raise ValueError("--raw-mae-scale must be greater than 0.")
    if args.raw_mae_weight < 0:
        raise ValueError("--raw-mae-weight must be greater than or equal to 0.")

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

    X_train, X_val, X_test, y_train, y_val, y_test, split_method = split_regression_data(
        X,
        y,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        stratified=not args.no_stratified_split,
    )
    print(
        f"Split sizes: train={len(X_train)}, val={len(X_val)}, test={len(X_test)} "
        f"({split_method})"
    )

    target_scaler = scalers.AsinhScaler(args.target_scale)
    scaled_y_train = target_scaler.scale(y_train)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = make_loader(
        X_train,
        scaled_y_train,
        y_train,
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
    ema_model = None
    if args.ema_decay > 0:
        ema_model = models.BoardInputRegression(X_train.shape[1]).to(device)
        configure_model_for_stability(
            ema_model,
            hidden_dropout=args.hidden_dropout,
            input_dropout=args.input_dropout,
            bn_momentum=args.bn_momentum,
        )
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

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
        args.prediction_clip,
    )
    if ema_model is not None:
        initial_ema_val_loss, initial_ema_val_rmse, initial_ema_val_mae = evaluate_regression_model(
            ema_model,
            val_loader,
            target_scaler,
            criterion,
            device,
            args.prediction_clip,
        )
    else:
        initial_ema_val_loss = float("nan")
        initial_ema_val_rmse = float("nan")
        initial_ema_val_mae = float("nan")

    initial_selected_val_mae = initial_val_mae
    best_source = "model"
    best_model = model
    if ema_model is not None and initial_ema_val_mae <= initial_val_mae:
        initial_selected_val_mae = initial_ema_val_mae
        best_source = "ema"
        best_model = ema_model

    best_val_mae = initial_selected_val_mae
    best_epoch = 0
    best_model_state = clone_model_state(best_model)
    epochs_since_best = 0

    progress_rows.append(
        {
            "epoch": 0,
            "lr": get_lr(optimizer),
            "train_loss": float("nan"),
            "train_scaled_loss": float("nan"),
            "train_raw_mae": float("nan"),
            "val_scaled_loss": initial_val_loss,
            "val_rmse": initial_val_rmse,
            "val_mae": initial_val_mae,
            "ema_val_scaled_loss": initial_ema_val_loss,
            "ema_val_rmse": initial_ema_val_rmse,
            "ema_val_mae": initial_ema_val_mae,
            "selected_val_mae": initial_selected_val_mae,
            "best_val_mae": initial_selected_val_mae,
            "best_source": best_source,
        }
    )
    print(f"Epoch [0/{args.epochs}], Val MAE: {initial_val_mae:.2f} cp")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            target_scaler=target_scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            amp_scaler=amp_scaler,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            loss_mode=args.loss,
            raw_mae_weight=args.raw_mae_weight,
            raw_mae_scale=args.raw_mae_scale,
            prediction_clip=args.prediction_clip,
            ema_model=ema_model,
            ema_decay=args.ema_decay,
        )

        val_loss, val_rmse, val_mae = evaluate_regression_model(
            model,
            val_loader,
            target_scaler,
            criterion,
            device,
            args.prediction_clip,
        )

        if ema_model is not None:
            ema_val_loss, ema_val_rmse, ema_val_mae = evaluate_regression_model(
                ema_model,
                val_loader,
                target_scaler,
                criterion,
                device,
                args.prediction_clip,
            )
        else:
            ema_val_loss = float("nan")
            ema_val_rmse = float("nan")
            ema_val_mae = float("nan")

        selected_model = model
        selected_val_mae = val_mae
        selected_source = "model"
        if ema_model is not None and ema_val_mae <= val_mae:
            selected_model = ema_model
            selected_val_mae = ema_val_mae
            selected_source = "ema"

        improved = selected_val_mae < best_val_mae - args.min_delta
        if improved:
            best_val_mae = selected_val_mae
            best_epoch = epoch
            best_source = selected_source
            best_model_state = clone_model_state(selected_model)
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        progress_rows.append(
            {
                "epoch": epoch,
                "lr": get_lr(optimizer),
                "train_loss": train_metrics["train_loss"],
                "train_scaled_loss": train_metrics["train_scaled_loss"],
                "train_raw_mae": train_metrics["train_raw_mae"],
                "val_scaled_loss": val_loss,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "ema_val_scaled_loss": ema_val_loss,
                "ema_val_rmse": ema_val_rmse,
                "ema_val_mae": ema_val_mae,
                "selected_val_mae": selected_val_mae,
                "best_val_mae": best_val_mae,
                "best_source": best_source,
            }
        )

        should_print = epoch == 1 or epoch % args.print_every == 0 or improved
        if should_print:
            marker = " *" if improved else ""
            ema_text = ""
            if ema_model is not None:
                ema_text = f"EMA MAE: {ema_val_mae:.2f} cp, "
            print(
                f"Epoch [{epoch}/{args.epochs}], "
                f"LR: {get_lr(optimizer):.2e}, "
                f"Train loss: {train_metrics['train_loss']:.4f}, "
                f"Train raw MAE: {train_metrics['train_raw_mae']:.2f} cp, "
                f"Val loss: {val_loss:.4f}, "
                f"Val RMSE: {val_rmse:.2f} cp, "
                f"Val MAE: {val_mae:.2f} cp, "
                f"{ema_text}"
                f"Best: {best_val_mae:.2f} cp @ {best_epoch} ({best_source}){marker}"
            )

        if epoch >= args.min_epochs and epochs_since_best >= args.patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best validation MAE was {best_val_mae:.2f} cp at epoch {best_epoch} "
                f"from {best_source} weights."
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
        args.prediction_clip,
    )
    print(
        f"Best epoch: {best_epoch} ({best_source}); "
        f"test loss: {test_loss:.4f}, test RMSE: {test_rmse:.2f} cp, "
        f"test MAE: {test_mae:.2f} cp"
    )

    metadata = {
        "args": json_safe_args(args),
        "device": str(device),
        "amp": use_amp,
        "split_method": split_method,
        "input_dim": X_train.shape[1],
        "model_class": "models.BoardInputRegression",
        "target_scaler": {
            "class": "scalers.AsinhScaler",
            "constant": args.target_scale,
        },
        "best_epoch": best_epoch,
        "best_source": best_source,
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
