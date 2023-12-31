import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from torch_geometric import loader
from time import time
import numpy as np
from numpy import random

def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False



df = pd.read_csv(r'src\tmp\approx_1p_ideal_comb_3times_new_ansatz.csv')

def ParseOp(string):
    string_list = list(string.strip('()').split("), ("))
    # how to remove the extra brackets
    string_list = list(map(lambda x: x.strip('('), string_list))
    string_list = list(map(lambda x: x.strip(')'), string_list))
    string_list = [tuple(map(int, sub.split(', '))) for sub in string_list]
    return string_list

# Prepare the graph data
num_nodes = 8
num_features = 4
num_output = 1
batch_size = 1

def Create_graph_data(index, op_col):
        edges = []
        x = torch.eye(num_nodes)
        op = ParseOp(op_col[index])
        for i in range(0, len(op)-1, 2):
            edges.append((op[i][0], op[i+1][0]))
            edges.append((op[i][1], op[i+1][1]))
                
        # Normalize edges to have smaller index first
        normalized_edges = [tuple(sorted(edge)) for edge in edges]

        # Count occurrences of each edge
        edge_counts = Counter(normalized_edges)

        # Prepare edge_index and edge_weight
        unique_edges = list(edge_counts.keys())
        edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor([edge_counts[edge] for edge in unique_edges], dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        
        return data


inputs = df[['Noisy_val_approx', 'sum_cx', 'sum_cx0', 'sum_t2']]
inputs = torch.tensor(inputs.values).reshape(-1, num_features)
inputs = inputs.type(torch.float32)
targets = df['Ideal_val']
print(targets.shape)
targets = torch.tensor(targets.values).reshape(-1, num_output)
targets = targets.type(torch.float32)
Operator_col = df['Operator']

# Create dataloader
dataset = [(Create_graph_data(i, Operator_col), inputs[i], targets[i]) for i in range(len(Operator_col))]
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# data_loader = loader.DataLoader(dataset, batch_size=batch_size, shuffle=True)
# random_seed(42, False)
IS_CUDA = torch.cuda.is_available()
DEVICE = 'cuda:0' if IS_CUDA else 'cpu'

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(8, 64)
        self.conv2 = GCNConv(64, 1) 


    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        # x = F.dropout(x, p=0.01)
        x = self.conv2(x, edge_index, edge_weight)
        return x

class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(num_features , 64)  
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, num_features)  
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.02)
        x = F.relu(self.fc2(x))
        # x = F.dropout(x, p=0.02)
        x = self.fc3(x)
        return x
    
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.gnn = GNN()
        # self.regressor = FCNN()
        self.fc1 = nn.Linear(num_nodes + num_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)


    def forward(self, x, edge_index, edge_weight, noisy_value):
        graph_out = self.gnn(x, edge_index, edge_weight)
        graph_out = graph_out.reshape(1, -1)
        
        # reg_out = self.regressor(noisy_value)
        # reg_out = torch.tensor(reg_out.detach().numpy()[0])
        

        noisy_input = noisy_value
        noisy_input = torch.tensor(noisy_input.detach().numpy()[0])
        combined = torch.cat((graph_out, noisy_input.unsqueeze(0)), dim=1)
        combined = self.fc1(combined)
        combined = F.relu(combined)
        # combined = F.dropout(combined, p=0.02)
        combined = self.fc2(combined)
        combined = F.relu(combined)
        # combined = F.dropout(combined, p=0.02)
        combined = self.fc3(combined)
        return combined

# Initialize the model
model = CombinedModel()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)



# Early stopping parameters
patience = 30 # How many epochs to wait after last time validation loss improved
best_loss = float('inf')
epochs_since_improvement = 0

# Training loop

for i in range(10):
    time0 = time()
    print('Iteration', i)
    for epoch in range(100):
        model = model.to(DEVICE)
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            target = torch.tensor(batch[2].detach().numpy()[0])
            target = target.reshape(1, -1)
            
            output = model(batch[0].x, batch[0].edge_index, batch[0].edge_attr, batch[1]) 
            # l1_lambda = 0.001  # The regularization parameter
            # l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = F.mse_loss(output, target) +1 #+ l1_lambda * l1_norm
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f'Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader)}')

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
            
                target = torch.tensor(batch[2].detach().numpy()[0])
                target = target.reshape(1, -1)
                
                output = model(batch[0].x, batch[0].edge_index, batch[0].edge_attr, batch[1]) 
                # print('predicted_value', output, 'true_value', target)
                # l1_lambda = 0.0001  # The regularization parameter
                # l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = F.mse_loss(output, target) #+ l1_lambda * l1_norm
        
                val_loss += loss.item()

                val_loss /= len(val_loader)
            # print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
        model.eval()
        # op_string = '(((0, 5), (2, 7)), ((1, 4), (3, 6)), ((0, 4), (2, 6)), ((1, 5), (3, 7)), ((0, 4), (3, 7)), ((1, 4), (2, 7)), ((4, 5), (6, 7)), ((0, 1), (2, 3)), ((1, 5), (2, 6)), ((0, 5), (3, 6)))'
        op_string = '(((0, 5), (2, 7)), ((1, 4), (3, 6)), ((1, 5), (3, 7)), ((0, 4), (2, 6)), ((1, 4), (2, 7)), ((0, 4), (3, 7)), ((0, 1), (2, 3)), ((4, 5), (6, 7)), ((1, 5), (2, 6)), ((0, 5), (3, 6)))'

        df_test = pd.DataFrame({
            'Operator': op_string,
            'Input': [-1.865049831834082]
        })

        op_col = df_test['Operator']
        noisy_val = torch.tensor([[-1.3067543819581207, 80/224,  60/224, 1]])
        test_graph = Create_graph_data(0, op_col)
        

        with torch.no_grad():
            graph_output = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr, noisy_val)
            print("Output of the GNN for a test graph:", graph_output)
        # Check for improvement
        # make directory
        
        # model_path = r'models\2times\early_stopping\best_model_2times.pt'
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_since_improvement = 0
            # Save the model if you want
            # torch.save(model.state_dict(), model_path)
        else:
            epochs_since_improvement += 1

    # if epochs_since_improvement >= patience:
    #     print(f"Early stopping triggered after {epoch+1} epochs")
    #     break

    # Test the model on final ansatz
    model.eval()
    # op_string = '(((0, 5), (2, 7)), ((1, 4), (3, 6)), ((0, 4), (2, 6)), ((1, 5), (3, 7)), ((0, 4), (3, 7)), ((1, 4), (2, 7)), ((4, 5), (6, 7)), ((0, 1), (2, 3)), ((1, 5), (2, 6)), ((0, 5), (3, 6)))'
    op_string = '(((0, 5), (2, 7)), ((1, 4), (3, 6)), ((1, 5), (3, 7)), ((0, 4), (2, 6)), ((1, 4), (2, 7)), ((0, 4), (3, 7)), ((0, 1), (2, 3)), ((4, 5), (6, 7)), ((1, 5), (2, 6)), ((0, 5), (3, 6)))'

    df_test = pd.DataFrame({
        'Operator': op_string,
        'Input': [-1.865049831834082]
    })


    op_col = df_test['Operator']
    noisy_val = torch.tensor([[-1.3067543819581207, 80/224,  60/224, 1]])
    test_graph = Create_graph_data(0, op_col)
    print(test_graph.edge_index)
    print(test_graph.edge_attr)
    # print('loss', loss_list) 

    # m = CombinedModel()
    # m = m.to(DEVICE)
    # m.eval()
    # m.load_state_dict(torch.load(model_path))

    with torch.no_grad():
        graph_output = model(test_graph.x, test_graph.edge_index, test_graph.edge_attr, noisy_val)
        print("Output of the GNN for a test graph:", graph_output)
    print('time taken', time() - time0)

# save model 
# g = graph_output.detach().numpy()
# g  = str(g)
# # remove element from string
# g = g.strip('[')
# g = g.strip(']')
# # remove element from string
# g = g.replace('-', '')
# g = g.replace('.', '')

# model_path = r'models\2times\model_2times_approx_noisy_wo_s_new_ansatz_' + g + '.pt'
# torch.save(model.state_dict(), model_path)

# load model
# m = CombinedModel()
# m = m.to(DEVICE)
# m.eval()
# m.load_state_dict(torch.load(model_path))

# with torch.no_grad():
#     graph_output = m(test_graph.x, test_graph.edge_index, test_graph.edge_attr, noisy_val)
#     print("Output of the GNN for a test graph:", graph_output)
