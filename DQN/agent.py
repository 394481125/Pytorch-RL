import random
from collections import deque
import torch
import torch.optim as optim
import numpy as np
from networks import *

class Agent:
    def __init__(self,state_size,action_size,bs,lr,tau,gamma,device):
        self.state_size=state_size
        self.action_size=action_size
        self.bs=bs  # batch size
        self.lr=lr
        self.tau=tau
        self.gamma=gamma
        self.device=device
        self.Q_local = DQN(self.state_size,self.action_size).to(self.device)
        self.Q_target = DQN(self.state_size, self.action_size).to(self.device)
        self.soft_update(1)
        self.optimizer = optim.Adam(self.Q_local.parameters(),self.lr)
        self.memory = deque(maxlen=100000)

    def soft_update(self,tau):
        # net2 new weight = tau * net1 weight +(1-tau) * net2 weight
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def act(self,state,eps=0):
        if(random.random())>eps:
            state = torch.tensor(state,dtype=torch.float32).to(self.device)
            with torch.no_grad(): # 这样不会生成计算图
                action_values = self.Q_local(state)
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self):
        experience = random.sample(self.memory,self.bs)
        states = torch.from_numpy(np.vstack([e[0] for e in experience])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experience])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experience])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experience])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experience]).astype(np.uint8)).float().to(self.device)

        Q_values = self.Q_local(states) # 预测当前状态的每个动作的奖励值
        Q_values = torch.gather(input=Q_values,dim=1,index=actions) # 根据表中动作的index获取需要的动作及其奖励值

        with torch.no_grad():
            Q_targets = self.Q_target(next_states) # 预测下一个状态每个动作的奖励
            Q_targets,_ = torch.max(input=Q_targets,dim=1,keepdim=True) # 找出最大奖励的动作奖励
            Q_targets = rewards + self.gamma * (1-dones) * Q_targets # 按比例加到奖励上，加上rewards和done信息

        loss = (Q_values-Q_targets).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()