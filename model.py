import torch
import torch.nn as nn # handling neuralnetwork layers
import torch.optim as optim # optimization
import torch.nn.functional as F # some functions
import os # for saving

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) #first linear layer
        self.linear2 = nn.Linear(hidden_size, output_size) #second layer

    def forward(self, x): 
        x = F.relu(self.linear1(x)) #need to apply linear layer,pass state to the first layer, apply activation function
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):  #i f folder doesnt exist its created
            os.makedirs(model_folder_path)
        
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma): # lr = learning rate
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # python optimixer state, using adam optimizer
        self.criterion = nn.MSELoss() # loss function

    def train_step(self, state, action, reward, next_state, done):
        # creating tensors
        state = torch.tensor(state, dtype=torch.float) 
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x) multiple values

        if len(state.shape) == 1:
            # only one dimension, but we want two (x, 1)
            state = torch.unsqueeze(state, 0) # converts to tuple
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) # defining tuple with one value

        # 1. predicted Q value with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done [idx]:
                # 2. Q_new = reward + gamma * max(next_predicted Q value) (bellman equation)=> only do this if not done
                Q_new= reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            
            target[idx][torch.argmax(action).item()] = Q_new # sets the index of one (from [0,0,1] for example) to the Q value

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred) #target = Q, pred = Qnew
        loss.backward()

        self.optimizer.step()
