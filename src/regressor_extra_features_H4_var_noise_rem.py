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

# Constants
seed = 3
torch.manual_seed(seed)
NUM_OUTPUT = 1
level = 5

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
    parser.add_argument('--model_path', type=str, default=config['model_path'], help='Path to save data')
    parser.add_argument('--bond_length', type=int, default=config['bond_length'], help='Bond length')
    parser.add_argument('--beta', type=int, default=config['beta'], help='Beta for huber loss')
    parser.add_argument('--data_path', type=str, default=config['data_path'], help='Path to save data')
    parser.add_argument('--molecule', type=str, default=config['molecule'], help='Molecule')
    
    return parser.parse_args()


# Model Classes


class Regressor(nn.Module):

    """Combined Graph and Fully Connected Neural Network model."""

    def __init__(self, num_features, hidden_dim):
        super(Regressor, self).__init__()
        
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, NUM_OUTPUT)

    def forward(self, noisy_value):
        
        noisy_input = noisy_value.detach().numpy()[0]
        combined = (torch.tensor(noisy_input).unsqueeze(0)) ####
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        return self.fc3(combined)

def train_and_evaluate(model, train_loader, val_loader, optimizer, epochs, beta, bond_length):

    """Train and evaluate the model."""

    best_loss = float('inf')

    for epoch in range(epochs):
        # Training Phase
        model.train()
        for batch in train_loader:

            optimizer.zero_grad()
            target = batch[1].reshape(1, -1)
            input = batch[0].reshape(1, -1)
            output = model(input)
            # loss = F.mse_loss(output, target)
            # huber loss
            loss = F.smooth_l1_loss(output, target, beta=beta) # beta = 0.4 for 1times
            # loss = F.mse_loss(output, target)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")   

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                target = batch[1].reshape(1, -1)
                input = batch[0].reshape(1, -1)
                output = model(input)
                val_loss += F.smooth_l1_loss(output, target, beta=beta).item()
                # val_loss += F.mse_loss(output, target).item()
            val_loss /= len(val_loader)
            

        # # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss

        # save best loss model 
            model_save_path =  'src/models/'+str(bond_length)+'times_regressor_extra_feat_dep_noise_level'+str(level)+'.pt'
            torch.save(model.state_dict(), model_save_path)
        
        data_path = 'src/tmp/var_noise/'
        molecule = 'H4'
        test_data_path = data_path + str(bond_length) + 'times_noise_test_data_' + molecule + '_dep_noise_level'+str(level)+ '.csv'
        test_data = pd.read_csv(test_data_path)
        test_point = torch.tensor(test_data[['Noisy_val_approx', 'two_qc_ratio', 'single_qc_ratio','singles_ratio', 'doubles_ratio', 'param_ratio']].values, dtype=torch.float32)
        test_model(model, test_point, bond_length)


def test_model(model, test_data, bond_length):

    """Tests the model on new data."""
    model = Regressor(num_features=6, hidden_dim=64)
    model.load_state_dict(torch.load('src/models/'+str(bond_length)+'times_regressor_extra_feat_dep_noise_level'+str(level)+'.pt'))
    model.eval()

    with torch.no_grad():
        output = model(test_data)
        # print(test_data.edge_index)
        # print(test_data.edge_attr)  
        print("Output of the GNN for a test graph:", output)


def save_model(model, filename):

    """Saves the model state."""

    torch.save(model.state_dict(), filename)

# Main execution
def main(args):

    num_features = args.num_features
    batch_size = args.batch_size
    train_test_split_ratio = args.train_test_split_ratio
    lr = args.learning_rate
    # weight_decay = args.weight_decay
    epochs = args.epochs
    model_path = args.model_path
    data_path = args.data_path
    molecule = args.molecule
    beta = args.beta
    bond_length = args.bond_length  
    num_nodes = args.num_nodes
    hidden_dim = args.hidden_dim
    hidden_channels = args.hidden_channels

    print('bond length is', bond_length)
    # Load and preprocess data
    # df = pd.read_excel('data/'+str(bond_length)+'times/neural_mitigation_data_wo_s_new_ansatz.xlsx')
    df = pd.read_csv('src/tmp/var_noise/'+str(bond_length)+'times_noise_train_data_H4_dep_noise_level'+str(level)+'.csv')
    # df = df[:469]
    # print(df['Operator'].tail())
    # df['REM_val'] = pd.to_numeric(df['REM_val'], errors='coerce') # this is an object; convert to float
    inputs = torch.tensor(df[['Noisy_val_approx', 'two_qc_ratio', 'single_qc_ratio','singles_ratio', 'doubles_ratio', 'param_ratio']].values, dtype=torch.float32)
    targets = torch.tensor(df['Seq_REM'].values.reshape(-1, 1), dtype=torch.float32)
    

    # Prepare dataset
    dataset = [(inputs[i], targets[i]) for i in range(len(targets))]
    train_size = int(train_test_split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
    # print('train loader', train_loader)
    val_loader = loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Regressor(num_features=num_features, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training and Evaluation
    train_and_evaluate(model, train_loader, val_loader, optimizer, epochs, beta, bond_length)

    # Test the model on new data
    test_data_path = data_path + str(bond_length) + 'times_noise_test_data_' + molecule + '_dep_noise_level'+str(level) +'.csv'
    test_data = pd.read_csv(test_data_path)
    test_point = torch.tensor(test_data[['Noisy_val_approx', 'two_qc_ratio', 'single_qc_ratio','singles_ratio', 'doubles_ratio', 'param_ratio']].values, dtype=torch.float32)
    test_model(model, test_point, bond_length)

    # Save the model
    # model_save_path = model_path + '25times_gnn_model_cnot.pt'
    # save_model(model, model_save_path)
    # print(f"Model saved at {model_save_path}")
    

if __name__ == '__main__':
    start_time = time()
    args = parse_arguments()
    main(args)
    print(f'Total time taken: {time() - start_time} seconds')

