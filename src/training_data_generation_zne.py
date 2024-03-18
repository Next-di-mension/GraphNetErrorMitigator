# Imports
import os
import pandas as pd
import itertools
import argparse
from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.fake_provider import FakeMelbourne, FakeGuadalupe, FakeRueschlikon, FakeSingapore, FakeTokyo
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer import AerSimulator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA
import warnings
from geometry_params import * 


from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.providers.fake_provider import *
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit_aer import AerSimulator

from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BackendEstimator, Estimator

from zne import zne, ZNEStrategy
from zne.noise_amplification import LocalFoldingAmplifier, GlobalFoldingAmplifier
from zne.extrapolation import *

import numpy as np

from qiskit.providers.fake_provider import FakeManila, FakeMelbourne, FakeGuadalupe
from qiskit_aer.noise import NoiseModel
import qiskit_aer.noise as noise
from scipy.optimize import minimize
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Constants and Globals
SEED = 170

def parse_arguments():
    parser = argparse.ArgumentParser(description="Quantum VQE Simulation")
    parser.add_argument(
        '--bond_length', 
        type=int, 
        default=2, 
        help='Bond length'
    )
    parser.add_argument(
        '--num_qubits',
        type=int,
        default=8,
        help='Number of qubits'
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
        default=3, 
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
        default='data/' ,
        help='Path to save data'
    )
    parser.add_argument(
        '--device_str',
        type=str,
        default='Melbourne',
        help='device to run on'
    )
    return parser.parse_args()

def select_device(device_str):
    if device_str == 'Melbourne':
        device = FakeMelbourne()
    elif device_str == 'Guadalupe':
        device = FakeGuadalupe()
    elif device_str == 'Rueschlikon':
        # device = FakeRueschlikon()
        device = FakeTokyo()
    else:
        raise ValueError('Invalid device string')
    return device

def initialize_estimators(shots, device_str):

    """Initializes noisy and noiseless quantum estimators.
    Args:
        shots (int): Number of shots for simulation.
        device_str (str): Device name
    Returns:
        noisy_estimator (AerEstimator): Noisy quantum estimator
        noiseless_estimator (AerEstimator): Noiseless quantum estimator
    """
    backend_fake = select_device(device_str) 
    coupling_map = backend_fake.configuration().coupling_map
    noise_model = NoiseModel.from_backend(backend_fake)
    # basis_gates = noise_model.basis_gates

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

def create_ansatz(exitations, parameters, num_qubits):

    """Creates a quantum circuit (ansatz) based on given parameters.
    Args:
        exitations (list): List of double excitations
        parameters (list): List of parameters
    Returns:
        qc (QuantumCircuit): Quantum circuit (ansatz)
    """

    qc = QuantumCircuit(num_qubits)
    hartree_fock(qc)
    for n in range(len(exitations)):
        if exitations[n][0][1] == 0 and  exitations[n][1][1] == 0:
            qc = single_excitation_circ(qc, exitations[n][0][0], exitations[n][1][0], parameters[n])
        else:
            qc = double_excitation_circ(qc, exitations[n][0][0], exitations[n][0][1], exitations[n][1][0], exitations[n][1][1], parameters[n])
    return qc


list_mitigated_result = []
list_noise_energy = []

def cost_func(device_str, shots, qubit_op, combinations, num_qubits, nuclear_repulsion_energy):
    """Return estimate of energy from estimator on noisy backend

    Returns:
        float: Energy estimate
    """
    def execution(parameters):
        '''Parameters:
        params (ndarray): Array of ansatz parameters
        '''     
        qc = create_ansatz(combinations, parameters, num_qubits)
        backend_fake = select_device(device_str) 
        noise_model_fake = NoiseModel.from_backend(backend_fake)
        basis_gates = noise_model_fake.basis_gates
        coupling_map = backend_fake.configuration().coupling_map


        backend = AerSimulator(noise_model=noise_model_fake,
                        coupling_map=coupling_map,
                        basis_gates=basis_gates)
        
        observable = qubit_op
        ZNEEstimator = zne(BackendEstimator)
        # ZNEEstimator = zne(Estimator(options={'shots': shots, 'seed_simulator': SEED}))
        # estimator = ZNEEstimator(backend=backend, options={'shots': shots, 'seed_simulator': SEED})
    
        estimator = ZNEEstimator(backend=backend, 
                                options={"seed": SEED, "shots": shots})

        zne_strategy = ZNEStrategy(
        noise_factors=[1,3,5],
        noise_amplifier=GlobalFoldingAmplifier(),
        extrapolator=LinearExtrapolator()
        )

        job = estimator.run(qc, observable, shots=shots, zne_strategy=zne_strategy)
        result = job.result()
        print(result)
        mitigated_energy = result.values
        
        noisy_energy = result.metadata[0]['zne']['noise_amplification']['values'][0]

        print("==============with noise model===========")
        print(f">>> Expectation value (Hartree): {noisy_energy + nuclear_repulsion_energy}")
        print(f">>> Total ground state energy (Hartree): {noisy_energy + nuclear_repulsion_energy}")
        print("=============================\n")

        print("=============mitigated==========")
        print(f">>> Expectation value (Hartree): {mitigated_energy + nuclear_repulsion_energy}")
        print(f">>> Total ground state energy (Hartree): {mitigated_energy + nuclear_repulsion_energy}")
        print("=============================\n")

        list_mitigated_result.append(mitigated_energy)
        list_noise_energy.append(noisy_energy)
            
        return mitigated_energy[0]
        
    return execution




def main(args):

    """Main function to execute the VQE algorithm and save results."""

    bond_length = args.bond_length
    molecule = args.molecule
    num_params = args.num_params
    max_iter = args.max_iter
    shots = args.shots
    data_path = args.data_path
    device_str = args.device_str
    num_qubits = args.num_qubits


    # Initialize ansatz
    key = molecule + '_'+ str(bond_length) + 'times' + '_' + device_str
    ansatz = Ansatz[key]
    print(f"{bond_length} times geometry")
    results = []
    
    # Initialize quantum operators and energies
    qubit_op = Observable[key]  
    nuclear_repulsion_energy = Energy_shift[key]  
    print('Nuclear Repulsion Energy: ', nuclear_repulsion_energy)
    print('Energy Shift: ', nuclear_repulsion_energy)

    for num_t2 in range(1, num_params+1):

        # Prepare combinations for ansatz
        T2_combinations = list(itertools.combinations(ansatz, num_t2))
        
        for combinations in T2_combinations:
            print('t2 noisy is ', combinations)
            
            observable = qubit_op
            list_noise_energy =[]
            initial_theta = [0.0]*num_t2

            vqe_result_noisy= minimize(
                    cost_func(device_str, shots, observable, combinations, num_qubits, nuclear_repulsion_energy),
                    x0=initial_theta,
                    method='COBYLA',
                    options={'maxiter':max_iter}   
                    )

            print(vqe_result_noisy)
            print('noisy energy values', list_noise_energy)
            print('mitigated energy values', list_mitigated_result) 

            if num_t2 == 1:
                combinations = combinations[0]

            optimal_energy_noisy = vqe_result_noisy
            results.append((combinations, optimal_energy_noisy)) 
           
    # Save results to CSV
    path = data_path+str(bond_length)+'times_noise_train_data_zne_' + device_str + '_' + molecule + '.csv'
    
    EM_data = pd.DataFrame(results, columns=['Operator', 'Noisy_val_approx'])
    EM_data.to_csv(path, mode='a', index=False)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
