import torch

from domineering_game import *
from zobrist_hashing import *

class DomineeringModel(torch.nn.Module):
    def __init__(self):
        super(DomineeringModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.l1 = torch.nn.Conv2d(1, 64, 3, stride=1, padding=1).to(self.device)
        self.l2 = torch.nn.Conv2d(64, 128, 3, stride=1, padding=1).to(self.device)
        self.eval_head = torch.nn.Linear(128*BOARD_SIZE*BOARD_SIZE, 1).to(self.device)
        self.policy_head = torch.nn.Linear(128*BOARD_SIZE*BOARD_SIZE, N_MOVES).to(self.device)

    def predict(self,representations):
        #representations is a torch tensor of shape (batch_size,1,BOARD_SIZE,BOARD_SIZE)
        representations = representations.to(self.device)
        x = torch.nn.functional.relu(self.l1(representations))
        x = torch.nn.functional.relu(self.l2(x))
        x = x.view(-1, 128*BOARD_SIZE*BOARD_SIZE)
        score = torch.sigmoid(self.eval_head(x))
        move_evals = torch.nn.functional.softmax(self.policy_head(x),dim=1)
        return [score,move_evals]
    
    def save(self,path):
        torch.save({
            'l1':self.l1.state_dict(),
            'l2':self.l2.state_dict(),
            'eval_head':self.eval_head.state_dict(),
            'policy_head':self.policy_head.state_dict()
        }, path)
    
    def load(self,path):
        checkpoint = torch.load(path)
        self.l1.load_state_dict(checkpoint['l1'])
        self.l2.load_state_dict(checkpoint['l2'])
        self.eval_head.load_state_dict(checkpoint['eval_head'])
        self.policy_head.load_state_dict(checkpoint['policy_head'])
        self.l1.to(self.device)
        self.l2.to(self.device)
        self.eval_head.to(self.device)
        self.policy_head.to(self.device)

    def train(self,optimizer,train_data,epochs=1,batch_size=32):
        #Loss function: binary cross-entropy for the evaluation head, and cross-entropy for the policy head
        #Weigh evaluation to be 5 times more important than policy
        #This is a hyperparameter that can be tuned
        states = train_data[0]
        moves = train_data[1]
        results = train_data[2]
        states = torch.tensor(np.array(states)).float().to(self.device)
        moves = torch.tensor(np.array(moves)).float().to(self.device)
        results = torch.tensor(np.array(results)).float().to(self.device)
        loss=0
        for epoch in range(epochs):
            for i in range(0,len(states),batch_size):
                optimizer.zero_grad()
                batch_states = states[i:i+batch_size]
                batch_moves = moves[i:i+batch_size]
                batch_results = results[i:i+batch_size]
                score,move_evals = self.predict(batch_states)
                eval_loss = torch.nn.functional.binary_cross_entropy(score.view(-1),batch_results)
                policy_loss = torch.nn.functional.cross_entropy(move_evals,batch_moves.argmax(1))
                loss = 5*eval_loss + policy_loss #the 5 is a hyperparameter
                loss.backward()
                optimizer.step()
        return loss.item()
    
    def copy(self):
        new_model = DomineeringModel()
        new_model.l1.load_state_dict(self.l1.state_dict())
        new_model.l2.load_state_dict(self.l2.state_dict())
        new_model.eval_head.load_state_dict(self.eval_head.state_dict())
        new_model.policy_head.load_state_dict(self.policy_head.state_dict())
        new_model.l1.to(self.device)
        new_model.l2.to(self.device)
        new_model.eval_head.to(self.device)
        new_model.policy_head.to(self.device)
        return new_model
    
