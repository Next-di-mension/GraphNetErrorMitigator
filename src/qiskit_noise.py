# Imports
import os
import pandas as pd
import itertools
import argparse
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.fake_provider import FakeMelbourne, FakeGuadalupe
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
import warnings
from geometry_params import *  # Assuming this is a custom module for your project

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants and Globals
SEED = 170
NUM_QUBITS = 8

def parse_arguments():
    parser = argparse.ArgumentParser(description="Quantum VQE Simulation")
    parser.add_argument(
        '--bond_length', 
        type=int, 
        default=3, 
        help='Bond length'
    )
    parser.add_argument(
        '--num_params', 
        type=int, 
        default=4, 
        help='Number of parameters'
    )
    parser.add_argument(
        '--max_iter', 
        type=int, 
        default=2000, 
        help='Maximum iterations for optimizer'
    )
    parser.add_argument(
        '--shots', 
        type=int, 
        default=30000, 
        help='Number of shots for simulation'
    )
    parser.add_argument(
        '--ansatz', 
        type=list, 
        default=Ansatz_3times_BH, 
        help='Ansatz for simulation'
    )

    parser.add_argument(
       '--data_path',
        type=str,
        default='src/tmp/' ,
        help='Path to save data'
    )
    return parser.parse_args()


def initialize_estimators(shots):

    """Initializes noisy and noiseless quantum estimators.
    Args:
        shots (int): Number of shots for simulation.
    Returns:
        noisy_estimator (AerEstimator): Noisy quantum estimator
        noiseless_estimator (AerEstimator): Noiseless quantum estimator
    """
    device = FakeGuadalupe()
    coupling_map = device.configuration().coupling_map
    noise_model = NoiseModel.from_backend(device)

    noisy_estimator = AerEstimator(
        backend_options={"method": "statevector", "coupling_map": coupling_map, "noise_model": noise_model},
        run_options={"seed": SEED, "shots": shots},
        transpile_options={"seed_transpiler": SEED},
        approximation=True
    )

    noiseless_estimator = AerEstimator(
        backend_options={"method": "statevector"},
        run_options={"seed": SEED, "shots": shots},
        transpile_options={"seed_transpiler": SEED},
        approximation=True
    )

    return noisy_estimator, noiseless_estimator

def hartree_fock(qc):
    """Applies Hartree-Fock initialization to the quantum circuit."""
    qc.x(0)
    qc.x(1)
    qc.x(4)
    qc.x(5)
    return qc

def double_excitation_circ(qc,i,j,k,l,theta):
    qc.cx(l,k)
    qc.cx(j,i)
    qc.cx(l,j)
    # 3 controlled ry
    qc.ry(theta/4, l)
    qc.cx(k,l,ctrl_state='0')
    qc.ry(-theta/4, l)
    qc.cx(i,l, ctrl_state='0')
    qc.ry(theta/4, l)
    qc.cx(k,l, ctrl_state='0')
    qc.ry(-theta/4, l)
    qc.cx(j,l)
    qc.ry(theta/4, l)
    qc.cx(k,l, ctrl_state='0')
    qc.ry(-theta/4, l)
    qc.cx(i,l, ctrl_state='0')
    qc.ry(theta/4, l)
    qc.cx(k,l, ctrl_state='0')
    qc.ry(-theta/4, l)
    qc.cx(j,l)

    qc.cx(l,j)
    qc.cx(j,i)
    qc.cx(l,k)

    return qc

def gate_count(qc, num_params):

    """Counts the number of gates in a quantum circuit."""

    cnots = qc.count_ops()['cx']  # CNOTs
    cx0 = 6*num_params  # CNOTs with control qubit 0
    # total gate count
    total = sum(qc.count_ops().values())
    return cnots, cx0, total

def create_ansatz(double_exit, parameters):

    """Creates a quantum circuit (ansatz) based on given parameters.
    Args:
        double_exit (list): List of double excitations
        parameters (list): List of parameters
    Returns:
        qc (QuantumCircuit): Quantum circuit (ansatz)
    """

    qc = QuantumCircuit(NUM_QUBITS)
    hartree_fock(qc)
    for n in range(len(double_exit)):
        double_excitation_circ(qc, double_exit[n][0][0], double_exit[n][0][1], double_exit[n][1][0], double_exit[n][1][1], parameters[n])
    return qc
    
def main(args):

    """Main function to execute the VQE algorithm and save results."""

    bond_length = args.bond_length
    num_params = args.num_params
    max_iter = args.max_iter
    shots = args.shots
    ansatz = args.ansatz
    data_path = args.data_path

    print(f"{bond_length} times geometry")
    results = []
    
    # Initialize quantum operators and energies
    qubit_op = op_2times  
    nuclear_repulsion_energy = nuc_2times  
    print('Nuclear Repulsion Energy: ', nuclear_repulsion_energy)
    print('Energy Shift: ', nuclear_repulsion_energy)

    # Initialize estimators
    noisy_estimator, noiseless_estimator = initialize_estimators(shots=shots)
    for num_t2 in range(1, num_params+1):
        # num_t2 = 4
        # num_params = 4  
        energy_list = []
        def store_intermediate_result(eval_count, parameters, mean, std):
            energy_list.append(mean+nuclear_repulsion_energy)
            print('Energy',mean+nuclear_repulsion_energy)
        # Create list of parameters for the ansatz
        param_list = [Parameter(f'phi{i}') for i in range(num_params)]

        # Prepare combinations for ansatz
        T2_combinations = list(itertools.combinations(ansatz, num_t2))
        
        for combinations in T2_combinations:
            qc = create_ansatz(combinations, param_list)
            qc_noisy = qc.copy()
            gate_counts = gate_count(qc, num_params)

            # Optimization and VQE execution noiseless
            print('t2 noiseless is ', combinations)
            optimizer = COBYLA(maxiter=max_iter)
            vqe = VQE(noiseless_estimator, qc, optimizer=optimizer, callback=store_intermediate_result, initial_point=[0.0]*num_t2)
            vqe_result = vqe.compute_minimum_eigenvalue(qubit_op)
            

            # Optimization and VQE execution noisy
            print('t2 noisy is ', combinations)
            optimizer_noisy = COBYLA(maxiter=max_iter)
            vqe_noisy = VQE(noisy_estimator, qc_noisy, optimizer=optimizer_noisy, callback=store_intermediate_result, initial_point=[0.0]*num_t2)
            vqe_result_noisy = vqe_noisy.compute_minimum_eigenvalue(qubit_op)

            if num_t2 == 1:
                combinations = combinations[0]
            
            cnot_count = gate_counts[0]/gate_counts[2]
            cnot_count0 = gate_counts[1]/gate_counts[2]

            optimal_energy = vqe_result.eigenvalue + nuclear_repulsion_energy
            optimal_energy_noisy = vqe_result_noisy.eigenvalue + nuclear_repulsion_energy
            results.append((combinations, optimal_energy_noisy, optimal_energy, cnot_count, cnot_count0, num_t2/10)) # num_params = no. of T2s

        # Save results to CSV
    path = data_path+str(bond_length)+'times_noise_data_Melbourne_BeH.csv'
    EM_data = pd.DataFrame(results, columns=['Operator', 'Noisy_val_approx', 'Ideal_val', 'sum_cx', 'sum_cx0', 'sum_t2'])
    EM_data.to_csv(path, mode='a', index=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
