import torch
from sklearn.metrics import f1_score
def get_output(data_loader,model):
    output = None
    target = None
    round = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model.forward(inputs)
            if round == 0:
                output = outputs
                round += 1
                target = targets
            else:
                output = torch.cat((output,outputs),dim=0)
                target = torch.cat((target,targets))
    return output,target

def test(loader, model):
    correct = total = 0
    predictions, targets = [], []

    with torch.no_grad():
        for inputs, target in loader:
            output = model.forward(inputs)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            predictions.extend(predicted.cpu().numpy())
            targets.extend(target.cpu().numpy().tolist())  

    accuracy = correct / total
    f1 = f1_score(targets, predictions, average='macro')
    return accuracy, f1