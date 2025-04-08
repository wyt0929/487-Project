import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.svm import SVC
import numpy as np
import torch.nn.functional as F
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        x = F.relu(self.fc1(out))
        return torch.softmax(self.fc2(x),dim=1)
    
    def train(self, epochs, train_loader):
        for epoch in range(epochs):
            for _, (inputs, targets) in enumerate(tqdm(train_loader, desc='Training', leave=False)):
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()


class SVMClassifier:
    def __init__(self, kernel='linear', C=1.0):
        self.model = SVC(kernel=kernel, C=C, probability=True)
    
    def forward(self, X):
        return torch.tensor(self.model.predict_proba(X))

    def train(self, train_loader):
        X_train, y_train = [], []
        for inputs, targets in tqdm(train_loader, desc='Training', leave=False):
            X_train.append(inputs)
            y_train.append(targets)
        X_train = np.vstack(X_train)  
        y_train = np.hstack(y_train)  
        self.model.fit(X_train, y_train)

    

class BertClassifier(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(BertClassifier, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, input):
        return torch.softmax(self.fc3(F.relu(self.fc2(F.relu(self.fc1(input))))),dim=1)
    
    def train(self, epochs, train_loader):
        for epoch in range(epochs):
            for _, (inputs, targets) in enumerate(tqdm(train_loader, desc='Training', leave=False)):
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

class MetaLearner(nn.Module):
    def __init__(self, num_classes=3):
        super(MetaLearner, self).__init__()
        self.fc1 = nn.Linear(9, 32)  
        self.fc2 = nn.Linear(32, num_classes)  
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    def forward(self, x):
        return torch.softmax(self.fc2(torch.relu(self.fc1(x))),dim=1) 

    def train_model(self, data, targets, epochs):
        for epoch in range(epochs):
            self.optimizer.zero_grad() 
            outputs = self(data)
            loss = self.criterion(outputs, targets)  
            loss.backward()  
            self.optimizer.step()  
            

class Actor(nn.Module):
    def __init__(self, num_classes=3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(9, 32)  
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    def forward(self, x):
        return torch.softmax(self.fc3(torch.relu((self.fc2(torch.relu(self.fc1(x)))))),dim=1) 
    
class Q(nn.Module):
    def __init__(self, num_classes=3):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(9, 32) 
        self.fc2 = nn.Linear(num_classes, 32) 
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    def forward(self, state, action):
        state_embedding = self.fc1(state)
        action_embedding = self.fc2(action)
        embedding = torch.cat((state_embedding,action_embedding),dim=1)
        output = torch.relu(self.fc4(torch.relu(self.fc3(embedding))))
        return output




