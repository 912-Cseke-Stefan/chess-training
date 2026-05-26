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

#print(X_train.shape[1])#, y_train.shape[1])
model = models.BoardInputBigClassifier(X_train.shape[1], 15).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

num_epochs = 500
loss_values = []
acc_values = []
#print(X_train[0])
#print(y_train[0])
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    y_pred = torch.argmax(y_pred, dim=1)

    loss_values.append(loss.item())
    acc_values.append(accuracy_fn(y_train, y_pred))
    
    model.eval()
    y_test_pred = model(X_test)
    y_test_pred = torch.argmax(y_test_pred, dim=1)
    #print(X_test.shape, y_test.shape, y_test_pred.shape)
    #accuracy_fn(y_train, y_test_pred)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy_fn(y_train, y_pred):.4f}, Test accuracy: {accuracy_fn(y_test, y_test_pred):.4f}')
