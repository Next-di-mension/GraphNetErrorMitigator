# Imports
import os
import pandas as pd
import itertools
import argparse
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.fake_provider import FakeMelbourne
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
        default=Ansatz_3times, 
        help='Ansatz for simulation'
    )

    parser.add_argument(
       '--data_path',
        type=str,
        default='src/tmp/' ,
        help='Path to save data'
    )
    return parser.parse_args()


def initialize_estimators(SHOTS):
    """Initializes noisy and noiseless quantum estimators."""
    device = FakeMelbourne()
    coupling_map = device.configuration().coupling_map
    noise_model = NoiseModel.from_backend(device)

    noisy_estimator = AerEstimator(
        backend_options={"method": "statevector", "coupling_map": coupling_map, "noise_model": noise_model},
        run_options={"seed": SEED, "shots": SHOTS},
        transpile_options={"seed_transpiler": SEED},
        approximation=True
    )

    noiseless_estimator = AerEstimator(
        backend_options={"method": "statevector"},
        run_options={"seed": SEED, "shots": SHOTS},
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

def gate_count(qc, NUM_PARAMS):
    """Counts the number of gates in a quantum circuit."""
    cnots = qc.count_ops()['cx']  # CNOTs
    cx0 = 6*NUM_PARAMS  # CNOTs with control qubit 0
    # total gate count
    total = sum(qc.count_ops().values())
    return cnots, cx0, total

def create_ansatz(double_exit, parameters):
    """Creates a quantum circuit (ansatz) based on given parameters."""
    qc = QuantumCircuit(NUM_QUBITS)
    hartree_fock(qc)
    for n in range(len(double_exit)):
        double_excitation_circ(qc, double_exit[n][0][0], double_exit[n][0][1], double_exit[n][1][0], double_exit[n][1][1], parameters[n])
    return qc
    
def main(args):
    """Main function to execute the VQE algorithm and save results."""

    BOND_LENGTH = args.bond_length
    NUM_PARAMS = args.num_params
    MAX_ITER = args.max_iter
    SHOTS = args.shots
    Ansatz = args.ansatz
    data_path = args.data_path

    print(f"{BOND_LENGTH} times geometry")
    results = []
    
    # Initialize quantum operators and energies
    qubit_op = op_3times  
    nuclear_repulsion_energy = nuc_3times  
    print('Nuclear Repulsion Energy: ', nuclear_repulsion_energy)
    print('Energy Shift: ', nuclear_repulsion_energy)

    # Initialize estimators
    noisy_estimator, noiseless_estimator = initialize_estimators(SHOTS=SHOTS)
    for num_t2 in range(1, NUM_PARAMS+1):
        energy_list = []
        def store_intermediate_result(eval_count, parameters, mean, std):
            energy_list.append(mean+nuclear_repulsion_energy)
            print('Energy',mean+nuclear_repulsion_energy)
        # Create list of parameters for the ansatz
        param_list = [Parameter(f'phi{i}') for i in range(NUM_PARAMS)]

        # Prepare combinations for ansatz
        T2_combinations = list(itertools.combinations(Ansatz, num_t2))
        
        for combinations in T2_combinations:
            qc = create_ansatz(combinations, param_list)
            qc_noisy = qc.copy()
            gate_counts = gate_count(qc, NUM_PARAMS)

            # Optimization and VQE execution noiseless
            optimizer = COBYLA(maxiter=MAX_ITER)
            vqe = VQE(noiseless_estimator, qc, optimizer=optimizer, callback=store_intermediate_result, initial_point=[0.0]*num_t2)
            vqe_result = vqe.compute_minimum_eigenvalue(qubit_op)
            

            # Optimization and VQE execution noisy
            optimizer_noisy = COBYLA(maxiter=MAX_ITER)
            vqe_noisy = VQE(noisy_estimator, qc_noisy, optimizer=optimizer_noisy, callback=store_intermediate_result, initial_point=[0.0]*num_t2)
            vqe_result_noisy = vqe_noisy.compute_minimum_eigenvalue(qubit_op)

            if num_t2 == 1:
                combinations = combinations[0]
            
            cnot_count = gate_counts[0]/gate_counts[2]
            cnot_count0 = gate_counts[1]/gate_counts[2]

            optimal_energy = vqe_result.eigenvalue + nuclear_repulsion_energy
            optimal_energy_noisy = vqe_result_noisy.eigenvalue + nuclear_repulsion_energy
            results.append((combinations, optimal_energy_noisy, optimal_energy, cnot_count, cnot_count0, num_t2)) # NUM_PARAMS = no. of T2s

        # Save results to CSV
    path = data_path+str(BOND_LENGTH)+'times_noise_data.csv'
    EM_data = pd.DataFrame(results, columns=['Operator', 'Noisy_val_approx', 'Ideal_val', 'sum_cx', 'sum_cx0', 'sum_t2'])
    EM_data.to_csv(path, mode='a', index=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
