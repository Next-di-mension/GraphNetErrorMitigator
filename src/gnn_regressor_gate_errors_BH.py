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
# Constants
NUM_NODES = 10
NUM_QUBITS = 10
NUM_OUTPUT = 1
l = []

def parse_arguments():
    parser = argparse.ArgumentParser(description="Graph Neural Network Training and Evaluation")
    parser.add_argument(
        '--num_features', 
        type=int, 
        default=4, 
        help='Number of features'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=1, 
        help='Batch size for training'
    )
    parser.add_argument(
        '--train_test_split_ratio', 
        type=float, 
        default=0.8, 
        help='Train/test split ratio'
    )
    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=0.001, 
        help='Learning rate for optimization'
    )
    parser.add_argument(
        '--weight_decay', 
        type=float, 
        default=5e-4, 
        help='Weight decay rate'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=150, 
        help='Number of training epochs'
    )
    parser.add_argument(
        '--patience', 
        type=int, 
        default=30, 
        help='Patience for early stopping'
    )
    parser.add_argument(
       '--model_path',
        type=str,
        default='src/models/',
        help='Path to save data'
    )
    parser.add_argument(
       '--bond_length',
        type=int,
        default=3,
        help='Bond length'
    )
    parser.add_argument(
       '--beta',
        type=int,
        default=0.008,
        help='Beta for huber loss'
    )
    return parser.parse_args()

# Functions


def gate_count(qc, num_params):

    """Counts the number of gates in a quantum circuit."""

    cnots = qc.count_ops()['cx']  # CNOTs
    cx0 = 6*num_params  # CNOTs with control qubit 0
    # total gate count
    total = sum(qc.count_ops().values())
    return cnots, cx0, total


cx_edges = []
def cnot_connectivity(qc):
    
    """Returns the connectivity of CNOT gates in a quantum circuit."""
    
    cx_edges = [(i.qubits[0].index,i.qubits[1].index) for i in qc if i.operation.name == 'cx']

    return cx_edges

            
def parse_operator_string(op_string):

    """Parses a string representing operators into a list of tuples."""

    string_list = [s.strip('()') for s in op_string.split("), (")]
    return [tuple(map(int, sub.split(', '))) for sub in string_list]

def cnot(bond_length):
    with open('src\cnot_connetivity_'+str(bond_length)+'times_BH.txt', 'r') as file:
        # Read lines from the file
        lines = file.readlines()

        # Iterate over the lines
        for line in lines:
            # Convert the string representation of the list to an actual list
            current_list = eval(line)
            l.append(current_list)
            # Now you can use the current_list in your loop
        
    return l

def create_graph_data(index, bond_length, gate_error: bool = True):

    """Creates graph data from the operator column for a given index.
    Args:
        index (int): Index of the operator column.
        operator_column (list): List of operator strings.
    Returns:
        Data object: Graph data for the given index.
    """

    # edges = []
    # x = torch.eye(NUM_NODES)
    if gate_error == True:
        x = ge.node_embd_mat
        x = torch.tensor(x, dtype=torch.float)
    else:
        x = torch.eye(NUM_NODES)
    # op = parse_operator_string(operator_column[index])
    # for i in range(0, len(op) - 1, 2):
        # edges.extend([(op[i][0], op[i + 1][0]), (op[i][1], op[i + 1][1])])
    edges = cnot(bond_length=bond_length)[index]
    edge_counts = Counter(tuple(sorted(edge)) for edge in edges)
    edge_index = torch.tensor(list(edge_counts.keys()), dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(list(edge_counts.values()), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)

# Model Classes
class GNN(nn.Module):

    """Graph Neural Network model."""

    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(NUM_NODES, 64)
        self.conv2 = GCNConv(64, 1)

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        return self.conv2(x, edge_index, edge_weight)

class CombinedModel(nn.Module):

    """Combined Graph and Fully Connected Neural Network model."""

    def __init__(self, num_features):
        super(CombinedModel, self).__init__()
        self.gnn = GNN()
        self.fc1 = nn.Linear(NUM_NODES + num_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, edge_index, edge_weight, noisy_value):
        graph_out = self.gnn(x, edge_index, edge_weight).reshape(1, -1)
        noisy_input = noisy_value.detach().numpy()[0]
        combined = torch.cat((graph_out, torch.tensor(noisy_input).unsqueeze(0)), dim=1)
        combined = F.relu(self.fc1(combined))
        combined = F.relu(self.fc2(combined))
        return self.fc3(combined)

def train_and_evaluate(model, train_loader, val_loader, optimizer, epochs, beta, bond_length):

    """Train and evaluate the model."""

    best_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            target = batch[2].reshape(1, -1)
            output = model(batch[0].x, batch[0].edge_index, batch[0].edge_attr, batch[1])
            # loss = F.mse_loss(output, target)
            # huber loss
            loss = F.smooth_l1_loss(output, target, beta=beta) # beta = 0.4 for 1times
            loss.backward()
            optimizer.step()
        
        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")   

        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                target = batch[2].reshape(1, -1)
                output = model(batch[0].x, batch[0].edge_index, batch[0].edge_attr, batch[1])
                val_loss += F.smooth_l1_loss(output, target, beta=beta).item()
            val_loss /= len(val_loader)
            

        # # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss

        # save best loss model 
            model_save_path =  'src/models/'+str(bond_length)+'times_gnn_model_cnot_val_huber_loss_ge_tr_gua_BH_2.pt'
            torch.save(model.state_dict(), model_save_path)
            # epochs_since_improvement = 0
        # else:
        #     epochs_since_improvement += 1
        #     if epochs_since_improvement == args.patience:
        #         print("Early Stopping!")
        #         break
         # Test the model on new data
        # noisy_val = torch.tensor([[-1.31598202227446, 80/224,  60/224, 1]])
        noisy_val = torch.tensor([[-24.46626978131897, 154/(154+136),  136/(154+136), 1]])
        test_graph_data = create_graph_data(-1,bond_length)  # Example test data
        test_model(model, test_graph_data, noisy_val, bond_length)


def test_model(model, test_data, circ_features, bond_length):

    """Tests the model on new data."""
    model = CombinedModel(num_features=4)
    model.load_state_dict(torch.load('src/models/'+str(bond_length)+'times_gnn_model_cnot_val_huber_loss_ge_tr_gua_BH_2.pt'))
    model.eval()

    with torch.no_grad():
        graph_output = model(test_data.x, test_data.edge_index, test_data.edge_attr, circ_features)
        # print(test_data.edge_index)
        # print(test_data.edge_attr)  
        print("Output of the GNN for a test graph:", graph_output)


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
    beta = args.beta
    bond_length = args.bond_length  


    # Load and preprocess data
    # df = pd.read_excel('data/'+str(bond_length)+'times/neural_mitigation_data_wo_s_new_ansatz.xlsx')
    df = pd.read_csv('src/tmp/'+str(bond_length)+'times_noise_data_Guadalupe_BH.csv')
    # df['REM_val'] = pd.to_numeric(df['REM_val'], errors='coerce') # this is an object; convert to float
    inputs = torch.tensor(df[['Noisy_val_approx', 'sum_2q_tr', 'sum_1q_tr', 'sum_t2']].values, dtype=torch.float32)
    targets = torch.tensor(df['Ideal_val'].values.reshape(-1, 1), dtype=torch.float32)
    operator_col = df['Operator']

    # Prepare dataset
    dataset = [(create_graph_data(i,bond_length), inputs[i], targets[i]) for i in range(len(operator_col))]
    train_size = int(train_test_split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = loader.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CombinedModel(num_features=num_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training and Evaluation
    train_and_evaluate(model, train_loader, val_loader, optimizer, epochs, beta, bond_length)

    # Test the model on new data
    # op_string = '(((0, 5), (2, 7)), ((1, 4), (3, 6)), ((1, 5), (3, 7)), ((0, 4), (2, 6)), ((1, 4), (2, 7)), ((0, 4), (3, 7)), ((0, 1), (2, 3)), ((4, 5), (6, 7)), ((1, 5), (2, 6)), ((0, 5), (3, 6)))'
    # test_data = pd.DataFrame({
    #         'Operator': op_string,
    #         'Input': [-1.8680172803812263]
    #     })

    # op_col = test_data['Operator']
    # noisy_val = torch.tensor([[-1.31598202227446, 80/224,  60/224, 1]])
    # noisy_val = torch.tensor([[-1.31598202227446, 80/224,  60/224, 1]])
    noisy_val = torch.tensor([[-24.46626978131897, 154/(154+136),  136/(154+136), 1]])
    test_graph_data = create_graph_data(-1,bond_length)  # Example test data
    test_model(model, test_graph_data, noisy_val, bond_length)

    # Save the model
    # model_save_path = model_path + '25times_gnn_model_cnot.pt'
    # save_model(model, model_save_path)
    # print(f"Model saved at {model_save_path}")
    

if __name__ == '__main__':
    start_time = time()
    args = parse_arguments()
    main(args)
    print(f'Total time taken: {time() - start_time} seconds')
