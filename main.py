import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from Dataset import LSTMDataset,BERTDataset,SVMDataset
from Models import LSTMModel,BertClassifier,MetaLearner,SVMClassifier,Actor,Q
from utils import get_output,test
from sklearn.metrics import f1_score
import numpy as np
import gc
import random
from collections import Counter
import time
import csv
import matplotlib.pyplot as plt
df = pd.read_csv('train.tsv', sep='\t')
df_train = df.head(len(df)//2)
svm_train = SVMDataset(df_train)
lstm_train = LSTMDataset(df_train)
bert_train = BERTDataset(df_train)
svm_train_loader = DataLoader(svm_train, batch_size=32, shuffle=True, collate_fn=lambda x: 
                        (torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True), 
                         torch.tensor([item[1] for item in x]))) 
lstm_train_loader = DataLoader(lstm_train, batch_size=32, shuffle=True, collate_fn=lambda x: 
                        (torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True), 
                         torch.tensor([item[1] for item in x]))) 
bert_train_loader = DataLoader(bert_train, batch_size=32, shuffle=True, collate_fn=lambda x: 
                        (torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True), 
                         torch.tensor([item[1] for item in x]))) 
lstm_model = LSTMModel(200,128,3)
bert_model = BertClassifier(768,3)
svm_model = SVMClassifier()
svm_model.train(svm_train_loader)
lstm_model.train(10,lstm_train_loader)
bert_model.train(100,bert_train_loader) 

del df_train, svm_train, lstm_train, bert_train, svm_train_loader, lstm_train_loader, bert_train_loader
gc.collect()

df_val = df.iloc[len(df)//2:]
svm_val = SVMDataset(df_val)
lstm_val = LSTMDataset(df_val)
bert_val = BERTDataset(df_val)
svm_val_loader = DataLoader(svm_val, batch_size=32, shuffle=True, collate_fn=lambda x: 
                        (torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True), 
                         torch.tensor([item[1] for item in x])))
lstm_val_loader = DataLoader(lstm_val, batch_size=32, shuffle=True, collate_fn=lambda x: 
                        (torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True), 
                         torch.tensor([item[1] for item in x])))
bert_val_loader = DataLoader(bert_val, batch_size=32, shuffle=True, collate_fn=lambda x: 
                        (torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True), 
                         torch.tensor([item[1] for item in x])))
bert_output,target = get_output(bert_val_loader,bert_model)
lstm_output,target = get_output(lstm_val_loader,lstm_model)
svm_output,target = get_output(svm_val_loader,svm_model)
meta_input = torch.cat((bert_output,lstm_output,svm_output),axis=1)
meta_learner = MetaLearner(3)
meta_learner.train_model(meta_input.float(),target,1)
torch.save(meta_input, 'val_input.pt')
torch.save(target, 'val_target.pt')

del df_val, svm_val, lstm_val, bert_val, svm_val_loader, lstm_val_loader, bert_val_loader
gc.collect()

df_test = pd.read_csv('test.tsv', sep='\t')
svm_test = SVMDataset(df_test)
lstm_test = LSTMDataset(df_test) 
bert_test = BERTDataset(df_test)
svm_test_loader = DataLoader(svm_test, batch_size=32, shuffle=False, collate_fn=lambda x: 
                         (torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True), 
                          torch.tensor([item[1] for item in x]))) 
lstm_test_loader = DataLoader(lstm_test, batch_size=32, shuffle=False, collate_fn=lambda x: 
                         (torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True), 
                          torch.tensor([item[1] for item in x]))) 
bert_test_loader = DataLoader(bert_test, batch_size=32, shuffle=False, collate_fn=lambda x: 
                         (torch.nn.utils.rnn.pad_sequence([item[0] for item in x], batch_first=True), 
                          torch.tensor([item[1] for item in x])))

meta_input = torch.load('val_input.pt', weights_only=True)
target = torch.load('val_target.pt', weights_only=True)
bert_accuracy, bert_f1 = test(bert_test_loader,bert_model)
print(bert_accuracy, bert_f1)
lstm_accuracy, lstm_f1 = test(lstm_test_loader,lstm_model)
print(lstm_accuracy, lstm_f1)
svm_accuracy, svm_f1 = test(svm_test_loader,svm_model)
print(svm_accuracy, svm_f1)


bert_output,target = get_output(bert_test_loader,bert_model)
lstm_output,target = get_output(lstm_test_loader,lstm_model)
svm_output,target = get_output(svm_test_loader,svm_model)


meta_input = torch.cat((bert_output,lstm_output,svm_output),axis=1)
meta_output = meta_learner.forward(meta_input.float())
torch.save(meta_input, 'test_input.pt')
torch.save(target, 'test_target.pt')
meta_input = torch.load('test_input.pt', weights_only=True)
target = torch.load('test_target.pt', weights_only=True)
_, predicted = torch.max(meta_output, 1)  
correct = (predicted == target).sum().item() 
total = target.size(0) 
accuracy = correct / total 
f1 = f1_score(target.cpu(), predicted.cpu(), average='macro')  
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')  
meta_input = torch.load('val_input.pt', weights_only=True)
target = torch.load('val_target.pt', weights_only=True)
actor = Actor(3)
q = Q(3)
rewards = []
times = []
for episode in range(5000):
    start_time = time.time()
    state = meta_input.float()
    action = actor.forward(state)
    weight1 = action[:,0].unsqueeze(1)
    weight2 = action[:,1].unsqueeze(1)
    weight3 = action[:,2].unsqueeze(1)
    model1_prediction = state[:,0:3]
    model2_prediction = state[:,3:6]
    model3_prediction = state[:,6:9]
    prediction = weight1*model1_prediction + weight2*model2_prediction + weight3*model3_prediction
    criterion = nn.CrossEntropyLoss(reduction='none')
    reward = 1 - criterion(prediction,target)
    for _ in range(40):
        q.optimizer.zero_grad()
        output = q(state, action)
        loss_q = q.criterion(output, reward.unsqueeze(1))
        loss_q.backward(retain_graph=True)
        q.optimizer.step()

    for _ in range(20):
        actor.optimizer.zero_grad()
        action = actor(state)  
        loss_actor = -q(state, action).mean()  
        loss_actor.backward(retain_graph=True)
        actor.optimizer.step()
    elapsed = time.time() - start_time
    avg_reward = reward.mean().item()
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1}: Reward={avg_reward:.4f}, Time={elapsed:.4f}s")
with open("training_rewards.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Episode", "Reward", "Time"])
    for i in range(len(rewards)):
        writer.writerow([i + 1, rewards[i], times[i]])
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Reward over Episodes")
plt.grid(True)
plt.savefig("reward_plot.png")
meta_input = torch.load('test_input.pt', weights_only=True)
target = torch.load('test_target.pt', weights_only=True)
state = meta_input.float()
action = actor.forward(state)
weight1 = action[:,0].unsqueeze(1)
weight2 = action[:,1].unsqueeze(1)
weight3 = action[:,2].unsqueeze(1)
model1_prediction = state[:,0:3]
model2_prediction = state[:,3:6]
model3_prediction = state[:,6:9]
prediction = weight1*model1_prediction + weight2*model2_prediction + weight3*model3_prediction
_, predicted = torch.max(prediction, 1)  
correct = (predicted == target).sum().item() 
total = target.size(0) 
accuracy = correct / total 
f1 = f1_score(target.cpu(), predicted.cpu(), average='macro')  
print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')

_, bert_predictions = torch.max(bert_output, dim=1)
_, lstm_predictions = torch.max(lstm_output, dim=1)
_, svm_predictions = torch.max(svm_output, dim=1)
prediction = []
for bert_prediction, lstm_prediction, svm_prediction in zip(bert_predictions, lstm_predictions, svm_predictions):
    votes = [bert_prediction.item(), lstm_prediction.item(), svm_prediction.item()]
    vote_counts = Counter(votes)
    most_common = vote_counts.most_common()

    if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
        chosen = most_common[0][0]
    else:
        tied_labels = []
        for label, count in most_common:
            if count == most_common[0][1]:
                tied_labels.append(label)
        chosen = random.choice(tied_labels)
    prediction.append(chosen)
prediction = torch.tensor(prediction)
correct = (prediction == target).sum().item() 
accuracy = correct / len(target)
f1 = f1_score(target.cpu(), prediction.cpu(), average='macro')

print(f'Accuracy: {accuracy:.4f}')
print(f'F1 Score: {f1:.4f}')







