# Imports
import os
import pandas as pd
import itertools
import argparse
from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.fake_provider import FakeMelbourne, FakeGuadalupe
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer import AerSimulator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
import warnings
from geometry_params import * 

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants and Globals
SEED = 170
NUM_QUBITS = 10

def parse_arguments():
    parser = argparse.ArgumentParser(description="Quantum VQE Simulation")
    parser.add_argument(
        '--bond_length', 
        type=int, 
        default=2, 
        help='Bond length'
    )
    parser.add_argument(
        '--molecule', 
        type=str, 
        default='H4', 
        help='Molecule'
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
    device = FakeMelbourne()
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
    qc.x(5)
    qc.x(6)
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

def single_excitation_circ(qc,i,k,theta):
    qc.cx(k,i)
    # controlled ry
    qc.ry(theta/2, k)
    qc.cx(i,k)
    qc.ry(-theta/2, k)
    qc.cx(i,k)

    qc.cx(k,i)

    return qc

def gate_count(qc, num_params):
    """Counts the number of gates in a quantum circuit."""
    
    backend_fake = FakeMelbourne()
    noise_model_fake = NoiseModel.from_backend(backend_fake)
    basis_gates = noise_model_fake.basis_gates
    coupling_map = backend_fake.configuration().coupling_map


    backend = AerSimulator(noise_model=noise_model_fake,
                        coupling_map=coupling_map,
                        basis_gates=basis_gates)
        
    for i in range(10):    
        tr_qc = transpile(qc, backend=backend, coupling_map=coupling_map,optimization_level=3,)
        # print(tr_qc)
        cnot_count = tr_qc.count_ops()['cx']  # CNOTs
        
        # total gate count
        total = (tr_qc.count_ops())#.values())
        print('total gate count is ', total)
        # print('cnot count is ', cnot_count)
        # print('single qubit gate count is ', total - cnot_count)
    return cnot_count, total

def create_ansatz(exitations, parameters):

    """Creates a quantum circuit (ansatz) based on given parameters.
    Args:
        exitations (list): List of double excitations
        parameters (list): List of parameters
    Returns:
        qc (QuantumCircuit): Quantum circuit (ansatz)
    """

    qc = QuantumCircuit(NUM_QUBITS)
    hartree_fock(qc)
    for n in range(len(exitations)):
        if exitations[n][0][1] == 0 and  exitations[n][1][1] == 0:
            qc = single_excitation_circ(qc, exitations[n][0][0], exitations[n][1][0], parameters[n])
        else:
            qc = double_excitation_circ(qc, exitations[n][0][0], exitations[n][0][1], exitations[n][1][0], exitations[n][1][1], parameters[n])
    return qc
    
def main(args):

    """Main function to execute the VQE algorithm and save results."""

    bond_length = args.bond_length
    molecule = args.molecule
    num_params = args.num_params
    max_iter = args.max_iter
    shots = args.shots
    data_path = args.data_path

    # Initialize ansatz
    key = molecule + '_'+ str(bond_length) + 'times'
    ansatz = Ansatz[key]

    print(f"{bond_length} times geometry")
    results = []
    
    # Initialize quantum operators and energies
    qubit_op = Observable[key]  
    nuclear_repulsion_energy = Energy_shift[key]  
    print('Nuclear Repulsion Energy: ', nuclear_repulsion_energy)
    print('Energy Shift: ', nuclear_repulsion_energy)

    # Initialize estimators
    noisy_estimator, noiseless_estimator = initialize_estimators(shots=shots)
    for num_t2 in range(1, num_params+1):
        num_t2 = 1
        num_params = 1  
        energy_list = []
        def store_intermediate_result(eval_count, parameters, mean, std):
            energy_list.append(mean+nuclear_repulsion_energy)
            print('Energy',mean+nuclear_repulsion_energy)
        # Create list of parameters for the ansatz
        # param_list = [Parameter(f'phi{i}') for i in range(num_params)]
        param_list = [0.123]

        # Prepare combinations for ansatz
        T2_combinations = list(itertools.combinations(ansatz, num_t2))
        
        for combinations in T2_combinations:
            print('t2 noiseless is ', combinations)
            qc = create_ansatz(combinations, param_list)
            qc_noisy = qc.copy()
            gate_counts = gate_count(qc, num_params)
            exit()
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
    exit()
    # Save results to CSV
    path = data_path+str(bond_length)+'times_noise_data_Melbourne_BeH.csv'
    EM_data = pd.DataFrame(results, columns=['Operator', 'Noisy_val_approx', 'Ideal_val', 'sum_cx', 'sum_cx0', 'sum_t2'])
    EM_data.to_csv(path, mode='a', index=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
