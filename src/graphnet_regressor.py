import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from torch_geometric import loader
from time import time
import argparse
import pandas as pd
from geometry_params import * 
import gate_errors as ge
import yaml

# seed = 3
# torch.manual_seed(seed)
NUM_OUTPUT = 1

def parse_arguments():

    parser = argparse.ArgumentParser(description="Graph Neural Network Training and Evaluation")
    
    # Parse the YAML file
    with open('src/config/gnn_config.yml', "r") as stream:
        config = yaml.safe_load(stream)

    parser.add_argument('--num_features', type=int, default=config['num_features'], help='Number of features')
    parser.add_argument('--hidden_channels', type=int, default=config['hidden_channels'], help='Number of hidden channels')
    parser.add_argument('--hidden_dim', type=int, default=config['hidden_dim'], help='Number of hidden dimensions')
    parser.add_argument('--num_nodes', type=int, default=config['num_nodes'], help='Number of nodes in the graph')
    parser.add_argument('--batch_size', type=int, default=config['batch_size'], help='Batch size for training')
    parser.add_argument('--train_test_split_ratio', type=float, default=config['train_test_split_ratio'], help='Train/test split ratio')
    parser.add_argument('--learning_rate', type=float, default=config['learning_rate'], help='Learning rate for optimization')
    parser.add_argument('--weight_decay', type=float, default=config['weight_decay'], help='Weight decay rate')
    parser.add_argument('--epochs', type=int, default=config['epochs'], help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=config['patience'], help='Patience for early stopping')
    parser.add_argument('--model_path', type=str, default=config['model_path'], help='Path to save model')
    parser.add_argument('--bond_length', type=int, default=config['bond_length'], help='Bond length')
    parser.add_argument('--beta', type=int, default=config['beta'], help='Beta for huber loss')
    parser.add_argument('--data_path', type=str, default=config['data_path'], help='Path to save data')
    parser.add_argument('--molecule', type=str, default=config['molecule'], help='Molecule')
    parser.add_argument('--machine', type=str, default=config['machine'], help='Machine')
    
    return parser.parse_args()


# Functions            
def parse_operator_string(op_string):

    """Parses a string representing operators into a list of tuples."""

    string_list = [s.strip('()') for s in op_string.split("), (")]
    return [tuple(map(int, sub.split(', '))) for sub in string_list]

def parse_cnot_connectivity(string):

    cnot_list = eval(string)
    return cnot_list

def create_graph_data(num_nodes, index, cnot_column, gate_error: bool = True):

    """Creates graph data from the operator column for a given index.
    Args:
        index (int): Index of the operator column.
        operator_column (list): List of operator strings.
    Returns:
        Data object: Graph data for the given index.
    """

    if gate_error == True:
        x = ge.node_embd_mat
        x = torch.tensor(x, dtype=torch.float)
    else:
        x = torch.eye(num_nodes, dtype=torch.float)

    edges = parse_cnot_connectivity(cnot_column[index])
    edge_counts = Counter(tuple(sorted(edge)) for edge in edges)
    edge_index = torch.tensor(list(edge_counts.keys()), dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(list(edge_counts.values()), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

# Model Classes
class GNN(nn.Module):

    """Graph Neural Network model."""

    def __init__(self, num_nodes, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_nodes, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, NUM_OUTPUT)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        return self.conv2(x, edge_index, edge_weight)

class CombinedModel(nn.Module):

    """Combined Graph and Fully Connected Neural Network model."""

    def __init__(self, num_features, num_nodes, hidden_dim):
        super(CombinedModel, self).__init__()
        self.gnn = GNN(num_nodes=num_nodes, hidden_channels=hidden_dim)
        self.fc1 = nn.Linear(num_nodes + num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, NUM_OUTPUT)

    def forward(self, x, edge_index, edge_weight, noisy_value):
        graph_out = self.gnn(x, edge_index, edge_weight).reshape(1, -1)
        noisy_input = noisy_value.detach().numpy()[0]
        combined = torch.cat((graph_out, torch.tensor(noisy_input).unsqueeze(0)), dim=1)
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        return self.fc3(combined)

def train_and_evaluate(model, train_loader, val_loader, optimizer, epochs, beta, bond_length, model_path):

    """Train and evaluate the model."""

    best_loss = float('inf')
    best_train_loss = float('inf')
    for epoch in range(epochs):
        # Training Phase
        model.train()
        for batch in train_loader:

            optimizer.zero_grad()
            target = batch[2].reshape(1, -1)
            output = model(batch[0].x, batch[0].edge_index, batch[0].edge_attr, batch[1])
        
            loss = F.smooth_l1_loss(output, target, beta=beta) 
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")   
        
        train_loss = loss.item()
        if train_loss < best_train_loss:
            best_train_loss = train_loss

            model_save_path = model_path +str(bond_length)+'times_gnn_model.pt'
            torch.save(model.state_dict(), model_save_path)


        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                target = batch[2].reshape(1, -1)
                output = model(batch[0].x, batch[0].edge_index, batch[0].edge_attr, batch[1])
                val_loss += F.smooth_l1_loss(output, target, beta=beta).item()
               
            val_loss /= len(val_loader)
            print(f"Validation Loss: {val_loss}")
            
        if val_loss < best_loss:
            best_loss = val_loss

            # save best loss model 
            model_save_path =  model_path+str(bond_length)+'times_gnn_bestloss_model.pt'
            torch.save(model.state_dict(), model_save_path)


def test_model(model, test_data, circ_features, bond_length, model_path, num_features, num_nodes, hidden_dim):

    """Tests the model on new data."""
    model = CombinedModel(num_features=num_features, num_nodes=num_nodes, hidden_dim=hidden_dim)
    model.load_state_dict(torch.load(model_path+str(bond_length)+'times_gnn_model.pt'))
    model.eval()

    with torch.no_grad():
        graph_output = model(test_data.x, test_data.edge_index, test_data.edge_attr, circ_features) 
        print("Output of the GNN for a test graph:", graph_output)

def test_best_train_model(model, test_data, circ_features, bond_length, model_path, num_features, num_nodes, hidden_dim):
    
        """Tests the model on new data."""
        model = CombinedModel(num_features=num_features, num_nodes=num_nodes, hidden_dim=hidden_dim)
        model.load_state_dict(torch.load(model_path+str(bond_length)+'times_gnn_bestloss_model.pt'))
        model.eval()
    
        with torch.no_grad():
            graph_output = model(test_data.x, test_data.edge_index, test_data.edge_attr, circ_features)  
            print("Output of the GNN for a test graph with best train loss:", graph_output)

def save_model(model, filename):

    """Saves the model state."""

    torch.save(model.state_dict(), filename)

# Main execution
def main(args):

    num_features = args.num_features
    batch_size = args.batch_size
    train_test_split_ratio = args.train_test_split_ratio
    lr = args.learning_rate
    epochs = args.epochs
    model_path = args.model_path
    data_path = args.data_path
    molecule = args.molecule
    beta = args.beta
    bond_length = args.bond_length  
    num_nodes = args.num_nodes
    hidden_dim = args.hidden_dim
    hidden_channels = args.hidden_channels
    machine = args.machine

    print('bond length is', bond_length)
    # Load the data
    df = pd.read_csv(data_path + str(bond_length)+'times_noise_train_data_' + machine +'_' + molecule + '.csv')
    inputs = torch.tensor(df[['Noisy_val_approx', 'two_qc_ratio', 'single_qc_ratio', 'param_ratio']].values, dtype=torch.float32)
    targets = torch.tensor(df['Ideal_val'].values.reshape(-1, 1), dtype=torch.float32)
    cnot_col = df['cnot_con']

    # Prepare dataset
    dataset = [(create_graph_data(num_nodes, i, cnot_column=cnot_col), inputs[i], targets[i]) for i in range(len(cnot_col))]
    train_size = int(train_test_split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    
    val_loader = loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Model setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CombinedModel(num_features=num_features, num_nodes=num_nodes, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training and Evaluation
    train_and_evaluate(model, train_loader, val_loader, optimizer, epochs, beta, bond_length, model_path)

    # Test the model on new data
    test_data_path = data_path + str(bond_length) + 'times_noise_test_data_' + machine +'_' +  molecule + '.csv'
    test_data = pd.read_csv(test_data_path)
    test_cnot_col = test_data['cnot_con']
    test_point = torch.tensor(test_data[['Noisy_val_approx', 'two_qc_ratio', 'single_qc_ratio', 'param_ratio']].values, dtype=torch.float32)
    test_graph_data = create_graph_data(num_nodes, 0, test_cnot_col)
    test_model(model, test_graph_data, test_point, bond_length, model_path, num_features, num_nodes, hidden_dim)
    test_best_train_model(model, test_graph_data, test_point, bond_length, model_path, num_features, num_nodes, hidden_dim)
    

if __name__ == '__main__':
    start_time = time()
    args = parse_arguments()
    main(args)
    print(f'Total time taken: {time() - start_time} seconds')

