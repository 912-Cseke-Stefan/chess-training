import data_preparation

X = []
y = []

with open("selected_top_level_games.csv", "r") as file:
    for line in file:
        if line[0] == 'F':
            continue
        
        line = line.strip()
        inputs, outputs = data_preparation.turn_fen_to_board_inputs_15_outputs(line)
        
        if inputs is None:
            continue
        
        X.append(inputs)
        y.append(outputs)

import torch
import models

if torch.cuda.is_available() is False:
    raise Exception("No GPU, no deal")
device = "cuda"

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y)#, dtype=torch.float)

#print(X.shape)
#print(y.shape)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.1, # 20% test, 80% train
                                                    random_state=43) # make the random split reproducible
#y_train = y_train.unsqueeze(1)   # this was not needed as I already have the outputs 
#y_test = y_test.unsqueeze(1)     # correctly structured as a list of lists
#print(y_train)
#y_train = y_train.unsqueeze(1)    # nevermind, CrossEntropyLoss expects of me to make the
#y_test = y_test.unsqueeze(1)      # abstraction that output should be one of C values and 
                                  # not an array of values for each of the C classes
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

from torch.utils.data import TensorDataset, DataLoader
import os

# Calculate class weights to handle imbalance - MORE AGGRESSIVE
class_counts = torch.bincount(y, minlength=15).float()
# Use sqrt of inverse frequency for less extreme but still meaningful weighting
class_weights = 1.0 / torch.sqrt(class_counts + 1)
class_weights = class_weights / class_weights.min()  # Scale so min weight is 1
class_weights = class_weights.to(device)
print(f'Class weights: {[f"{w:.2f}" for w in class_weights.cpu().tolist()]}')

model = models.BoardInputBigClassifier(X_train.shape[1], 15).to(device)

# Load model to continue training
if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth'))
    print('Loaded existing model from model.pth')

criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

batch_size = 2048
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 20
loss_values = []
acc_values = []
best_test_acc = 0.0

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

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), 'model.pth')

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')

print(f'Best test accuracy: {best_test_acc:.4f}')
