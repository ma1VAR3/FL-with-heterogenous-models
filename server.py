import torch
import torch.nn as nn
import torch.nn.functional as F


class ServerNet(nn.Module):
    def __init__(self):
        super(ServerNet, self).__init__()
        self.cv1 = nn.Conv2d(3, 32, 5)
        self.cv2 = nn.Conv2d(32, 64, 5)
        self.cv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.cv1(x))
        x = self.pool(F.relu(self.cv2(x)))
        x = self.pool(F.relu(self.cv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Server:
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=32, shuffle=True
        )
        self.model = ServerNet()

    def aggregate(self, client_updates):
        for cu in client_updates:
            for key in cu:
                cu[key] *= 1 / len(client_updates)
        self.model.load_state_dict(client_updates[0])
        for cu in client_updates[1:]:
            for key in cu:
                self.model.state_dict()[key] += cu[key]

    def federate(self, clients):
        updates = []
        for c in clients:
            print("\n -----> Training client {}".format(clients.index(c)))
            c_model = ServerNet()
            c_model.load_state_dict(self.model.state_dict())
            c.train(5, c_model)
            updates.append(c.get_weights())
        self.aggregate(updates)

    def evaluate(self):
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(
            "========== !!!!! Accuracy of the server model on test images: %d %% !!!!! =========="
            % (100 * correct / total)
        )
