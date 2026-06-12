import data_preparation

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


X, y = read_from_file("selected_top_level_games.csv", data_preparation.turn_fen_to_board_inputs_15_outputs)

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

def train_big_classifier():
    model = models.BoardInputBigClassifier(X_train.shape[1], 15).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    batch_size = 512
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 300
    loss_values = []
    acc_values = []

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
        loss_values.append(epoch_loss)
        acc_values.append(train_acc)

        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            y_test_pred = torch.argmax(y_test_pred, dim=1)
            test_acc = accuracy_fn(y_test, y_test_pred)

        scheduler.step(test_acc)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')


def train_regression_model():
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

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.7, weight_decay=1e-4, nesterov=True)
    scaler = scalers.TanhScaler(5000)

    scaled_y_train = scaler.scale(y_train_regression).view(-1, 1)
    y_test_regression = y_test_regression.view(-1, 1)

    batch_size = 256
    train_dataset = TensorDataset(X_train_regression, scaled_y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 300
    loss_values = []
    test_loss_values = []

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

        model.eval()
        with torch.no_grad():
            scaled_y_test_pred = model(X_test_regression)

            # clamping is necessary because there is no guarantee the model will respect
            # the boundaries of the output of tanh
            scaled_y_test_pred = torch.clamp(scaled_y_test_pred, -0.999999, 0.999999)

            y_test_pred = scaler.reverse(scaled_y_test_pred)
            test_loss = criterion(y_test_pred, y_test_regression).item()

        test_loss_values.append(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train scaled loss: {epoch_loss:.4f}, Test loss: {test_loss:.4f}')
	
