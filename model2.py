import torch
import torch.nn as nn #handling neuralnetwork layers
import torch.optim as optim #optimization
import torch.nn.functional as F #some functions
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) #first layer of the network
        self.linear2 = nn.Linear(hidden_size, output_size) #second layer

    def forward(self, x):
        x = F.relu(self.linear1(x)) #pass state to the first layer
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma): # lr = learning rate
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # creating tensors
        state = torch.tensor(state, dtype=torch.float) 
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x) multiple values?

        if len(state.shape) == 1:
            # (1, x) one value i guesss?
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # defining tuple with one value

        # 1. predicted Q values with current state
        pred = self.model(state) # = = or =?

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done [idx]:
                Q_new= reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action).item()] = Q_new

        # 2. Q_new = reward + gamma * max(next_predicted Q value) => only do this if not done 
        # pred.clone()
        # preds[argmax(action)] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
