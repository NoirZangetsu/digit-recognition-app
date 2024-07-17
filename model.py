import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ImprovedDigitRecognitionCNN(nn.Module):
    def __init__(self):
        super(ImprovedDigitRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.best_loss = float('inf')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def update(self, image, true_label):
        self.train()
        self.optimizer.zero_grad()
        output = self(image)
        loss = F.nll_loss(output, true_label.unsqueeze(0))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def is_best(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            return True
        return False

    def save_best(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'best_loss': self.best_loss
        }, path)

    def load_best(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.best_loss = checkpoint['best_loss']