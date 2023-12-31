# Imports
import os
import pandas as pd
import itertools
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
BOND_LENGTH = 3
NUM_PARAMS = 1
SEED = 170
NUM_QUBITS = 8
MAX_ITER = 2000
SHOTS = 30000

# Create a directory to store results
os.makedirs(r'src\tmp', exist_ok=True)

def initialize_estimators():
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


def gate_count(qc):
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

def main():
    """Main function to execute the VQE algorithm and save results."""
    print(f"{BOND_LENGTH} times geometry")

    # Initialize quantum operators and energies
    qubit_op = op_3times  # Assuming defined in geometry_params
    nuclear_repulsion_energy = nuc_3times  # Assuming defined in geometry_params
    print('Nuclear Repulsion Energy: ', nuclear_repulsion_energy)
    print('Energy Shift: ', nuclear_repulsion_energy)

    # Initialize estimators
    noisy_estimator, noiseless_estimator = initialize_estimators()

    # Create list of parameters for the ansatz
    param_list = [Parameter(f'phi{i}') for i in range(NUM_PARAMS)]

    # Prepare combinations for ansatz
    all_T2 = Ansatz_3times  # Assuming defined in geometry_params
    list_T2_combinations = list(itertools.combinations(all_T2, NUM_PARAMS))

    results = []
    for list_T2 in list_T2_combinations:
        qc = create_ansatz(list_T2, param_list)
        gate_counts = gate_count(qc)

        # Optimization and VQE execution noiseless
        optimizer = COBYLA(maxiter=MAX_ITER)
        vqe = VQE(noiseless_estimator, qc, optimizer=optimizer)
        vqe_result = vqe.compute_minimum_eigenvalue(qubit_op)
        

        # Optimization and VQE execution noisy
        optimizer_noisy = COBYLA(maxiter=MAX_ITER)
        vqe_noisy = VQE(noisy_estimator, qc, optimizer=optimizer_noisy)
        vqe_result_noisy = vqe_noisy.compute_minimum_eigenvalue(qubit_op)

        if NUM_PARAMS == 1:
            list_T2 = list_T2[0]
        
        cnot_count = gate_counts[0]/gate_counts[2]
        cnot_count0 = gate_counts[1]/gate_counts[2]

        # Store results
        optimal_energy = vqe_result.eigenvalue + nuclear_repulsion_energy
        optimal_energy_noisy = vqe_result_noisy.eigenvalue + nuclear_repulsion_energy
        results.append((list_T2, optimal_energy_noisy, optimal_energy, cnot_count, cnot_count0, NUM_PARAMS)) # NUM_PARAMS = no. of T2s


    # Save results to CSV
    EM_data = pd.DataFrame(results, columns=['Operator', 'Noisy_val_approx', 'Ideal_value', 'sum_cx', 'sum_cx0', 'sum_t2'])

    EM_data.to_csv(f'src/tmp/approx_{NUM_PARAMS}p_ideal_comb_{BOND_LENGTH}times_new_ansatz.csv', index=False)

if __name__ == '__main__':
    main()