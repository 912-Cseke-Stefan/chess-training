import data_preparation
from pathlib import Path
from datetime import datetime

def read_from_file(filename: str, output_parser):
    X = []
    y = []

    with open(filename, "r") as file:
        for line in file:
            if line[0] == 'F' or line[0] == 'f':
                continue
            
            line = line.strip()
            inputs, outputs = output_parser(line)
            
            if inputs is None:
                continue
            
            X.append(inputs)
            y.append(outputs)
    
    return X, y

import torch
import models
import scalers

if torch.cuda.is_available() is False:
    raise Exception("No GPU, no deal")
device = "cuda"


X, y = read_from_file("selected_top_level_games.csv", data_preparation.turn_fen_to_board_inputs_20_outputs)

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, # 20% test, 80% train
                                                    random_state=43) # make the random split reproducible

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

from torch.utils.data import TensorDataset, DataLoader

def get_timestamped_artifact_paths(model_filename, finished_at):
    save_dir = Path("save")
    model_filename = Path(model_filename)
    model_stem = model_filename.stem
    model_suffix = model_filename.suffix or ".pth"

    model_path = save_dir / f"{model_stem}_{finished_at}{model_suffix}"
    progress_path = save_dir / f"progres_{model_stem}_{finished_at}.txt"

    return model_path, progress_path

def clone_model_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }

def save_progress_values(progress_path, metric_name, values):
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    with open(progress_path, "w") as file:
        file.write(f"{metric_name}\n")
        for epoch, value in enumerate(values, start=1):
            file.write(f"{epoch},{value}\n")

def save_training_artifacts(model, model_state, progress_values, model_filename, metric_name):
    finished_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path, progress_path = get_timestamped_artifact_paths(model_filename, finished_at)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_state is None:
        model_state = clone_model_state(model)

    torch.save(model_state, model_path)
    save_progress_values(progress_path, metric_name, progress_values)

    print("-------------------------------")
    print(f"Saved final model to {model_path}")
    print(f"Saved progress to {progress_path}")

def remember_classifier_model_if_better(
    model,
    current_performance,
    best_stored_performance,
    epoch,
    best_model_state,
    min_epochs=50
):
    if epoch + 1 < min_epochs:
        return best_stored_performance, best_model_state

    if current_performance <= best_stored_performance:
        return best_stored_performance, best_model_state

    print(f"  Remembered classifier model at epoch {epoch + 1} with test accuracy {current_performance:.4f}")

    return current_performance, clone_model_state(model)

def remember_regression_model_if_better(
    model,
    current_test_mae,
    best_stored_test_mae,
    epoch,
    best_model_state,
    min_epochs=50
):
    if epoch + 1 < min_epochs:
        return best_stored_test_mae, best_model_state

    if current_test_mae >= best_stored_test_mae:
        return best_stored_test_mae, best_model_state

    print(f"  Remembered regression model at epoch {epoch + 1} with test MAE {current_test_mae:.2f} cp")

    return current_test_mae, clone_model_state(model)

def train_big_classifier():
    model_filename = "big_classifier_20.pth"
    #                                         magic constant vv
    model = models.BoardInputBigClassifier(X_train.shape[1], 20).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    batch_size = 256
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 300
    acc_values = []
    best_stored_test_acc = float("-inf")
    best_model_state = None

    # value before any training
    model.eval()
    with torch.no_grad():
        y_test_pred = model(X_test)
        y_test_pred = torch.argmax(y_test_pred, dim=1)
        test_acc = accuracy_fn(y_test, y_test_pred)

    acc_values.append(test_acc)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            correct += torch.eq(y_batch, torch.argmax(y_pred, dim=1)).sum().item()
            total += X_batch.size(0)

        epoch_loss /= total
        train_acc = (correct / total) * 100
        
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            y_test_pred = torch.argmax(y_test_pred, dim=1)
            test_acc = accuracy_fn(y_test, y_test_pred)

        scheduler.step(test_acc)
        acc_values.append(test_acc)

        best_stored_test_acc, best_model_state = remember_classifier_model_if_better(
            model,
            test_acc,
            best_stored_test_acc,
            epoch,
            best_model_state
        )

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')

    save_training_artifacts(
        model,
        best_model_state,
        acc_values,
        model_filename,
        "test_accuracy"
    )

def evaluate_regression_model(model, X, y_raw, scaler, criterion, batch_size=4096):
    metric_dataset = TensorDataset(X, y_raw.view(-1, 1))
    metric_loader = DataLoader(metric_dataset, batch_size=batch_size, shuffle=False)

    scaled_loss_total = 0
    squared_error_total = 0
    absolute_error_total = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in metric_loader:
            scaled_y_pred = model(X_batch)
            scaled_y_batch = scaler.scale(y_batch)
            y_pred = scaler.reverse(scaled_y_pred)
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

def train_regression_model():
    model_filename = "regression_big_adamw_best.pth"
    X_regression, y_regression = read_from_file(
        "selected_top_level_games.csv",
        data_preparation.turn_fen_to_board_inputs_raw_output
    )

    X_regression = torch.tensor(X_regression, dtype=torch.float)
    y_regression = torch.tensor(y_regression, dtype=torch.float)

    X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(
        X_regression,
        y_regression,
        test_size=0.2,
        random_state=43
    )

    X_train_regression = X_train_regression.to(device)
    X_test_regression = X_test_regression.to(device)
    y_train_regression = y_train_regression.to(device)
    y_test_regression = y_test_regression.to(device)

    model = models.BoardInputRegression(X_train_regression.shape[1]).to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.7, weight_decay=0, nesterov=True)
    scaler = scalers.AsinhScaler(1000)
    #scaler = scalers.TanhScaler(1000)
    #scaler = scalers.DistribScaler(y_train_regression.mean(), y_train_regression.std())
    #print(y_train_regression.mean(), y_train_regression.std())
    
    y_train_regression = y_train_regression.view(-1, 1)
    y_test_regression = y_test_regression.view(-1, 1)
    scaled_y_train = scaler.scale(y_train_regression)

    batch_size = 256
    train_dataset = TensorDataset(X_train_regression, scaled_y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 300
    print_every = 10
    loss_values = []
    test_rmse_values = []
    test_mae_values = []
    best_stored_test_mae = float("inf")
    best_model_state = None

    # value before any training
    test_scaled_loss, test_rmse, test_mae = evaluate_regression_model(
        model,
        X_test_regression,
        y_test_regression,
        scaler,
        criterion
    )

    test_mae_values.append(test_mae)


    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * X_batch.size(0)
            total += X_batch.size(0)

        epoch_loss /= total
        loss_values.append(epoch_loss)

        test_scaled_loss, test_rmse, test_mae = evaluate_regression_model(
            model,
            X_test_regression,
            y_test_regression,
            scaler,
            criterion
        )
        test_rmse_values.append(test_rmse)
        test_mae_values.append(test_mae)

        best_stored_test_mae, best_model_state = remember_regression_model_if_better(
            model,
            test_mae,
            best_stored_test_mae,
            epoch,
            best_model_state
        )

        if (epoch + 1) % print_every == 0:
            train_scaled_loss, train_rmse, train_mae = evaluate_regression_model(
                model,
                X_train_regression,
                y_train_regression,
                scaler,
                criterion
            )

            print(
                f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train fit loss: {epoch_loss:.4f}, '
                f'Train eval loss: {train_scaled_loss:.4f}, '
                f'Train RMSE: {train_rmse:.2f} cp, '
                f'Train MAE: {train_mae:.2f} cp, '
                f'Test eval loss: {test_scaled_loss:.4f}, '
                f'Test RMSE: {test_rmse:.2f} cp, '
                f'Test MAE: {test_mae:.2f} cp'
            )

    save_training_artifacts(
        model,
        best_model_state,
        test_mae_values,
        model_filename,
        "test_mae"
    )


#train_big_classifier()
train_regression_model()
