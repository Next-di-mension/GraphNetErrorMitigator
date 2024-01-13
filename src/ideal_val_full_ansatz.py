# imports
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.providers.fake_provider import *
from qiskit.providers.fake_provider import FakeMelbourne, FakeGuadalupe
from qiskit_aer.noise import NoiseModel
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA,COBYLA, NELDER_MEAD, SLSQP, ADAM, L_BFGS_B
import warnings
from geometry_params import *

warnings.filterwarnings('ignore')
molecule = 'BH'
bond_length = 3 
key = molecule + '_'+ str(bond_length) + 'times'
ansatz = Ansatz[key]
print(f"{bond_length} times geometry")
results = []
    
# Initialize quantum operators and energies
qubit_op = Observable[key]  
nuclear_repulsion_energy = Energy_shift[key]  
shift = nuclear_repulsion_energy

print('Nuclear Repulsion Energy: ', nuclear_repulsion_energy)
print('Energy Shift: ',shift)


device = FakeGuadalupe()
coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)
seed = 170


noisy_estimator = AerEstimator(
    backend_options={
        "method": "statevector",
        "coupling_map": coupling_map,
        "noise_model": noise_model,
    },
    run_options={"seed":seed, "shots": 30000},
    transpile_options={"seed_transpiler": seed},
    approximation=True
)

noiseless_estimator = AerEstimator(
    backend_options={
        "method": "statevector",
    },
    run_options={"seed": seed, "shots": 30000},
    transpile_options={"seed_transpiler": seed},
    approximation = True
)

num_params = 11
param_list = list()

for i in range(num_params):
    globals()['phi{}'.format(i)] = Parameter('phi{}'.format(i))
    param_list.append(globals()['phi{}'.format(i)])


def Hartree_Fock(qc):
    qc.x(0)
    qc.x(1)
    qc.x(5)
    qc.x(6)
    return qc

def first_excitation_circ(qc,i,k,theta):
    qc.cx(k,i)
    # controlled ry
    qc.ry(theta/2, k)
    qc.cx(i,k)
    qc.ry(-theta/2, k)
    qc.cx(i,k)

    qc.cx(k,i)

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

# list_T1=[((0), (2)), ((1), (3)), ((4), (6)), ((5), (7))]

final_ansatz_wo_s = ansatz

list_T2 = final_ansatz_wo_s
T2_angle = param_list
num_qubits = 10

qc = QuantumCircuit(num_qubits)
Hartree_Fock(qc)
for n in range(len(list_T2)):
    double_excitation_circ(qc, list_T2[n][0][0], list_T2[n][0][1], list_T2[n][1][0], list_T2[n][1][1], T2_angle[n])
# for m in range(len(list_T1)):
#     first_excitation_circ(qc, list_T1[m][0], list_T1[m][1], T2_angle[m+len(list_T2)])

values1 = list()
def store_intermediate_result(eval_count, parameters, mean, std):
    energy_list.append(mean+nuclear_repulsion_energy)
    print('Energy',mean+nuclear_repulsion_energy)



optimizer=COBYLA(maxiter=20000)

avg_list = list()
opt_energy_list = list()
opt_pt_list = list()


for i in range(1):
    energy_list = list()
    vqe = VQE(noisy_estimator, qc, optimizer=optimizer, callback=store_intermediate_result, initial_point=[0.0]*num_params)
    vqe_result = vqe.compute_minimum_eigenvalue(qubit_op)
    sum1 = 0.0
    num1 = 0.0
    opt_pt_list.append(vqe_result.optimal_point[0])
    opt_energy_list.append(vqe_result.eigenvalue+nuclear_repulsion_energy)
    length=len(energy_list)
    for j in range(length-1,length-5,-1):
        sum1=sum1+energy_list[j]
        num1=num1+1

    avg_list.append(sum1/num1)    
    
    print('================================================================================================================')
    print('avg energy',sum(avg_list)/len(avg_list))
    print('Opt_Points',opt_pt_list)
    print('================================================================================================================')
    

# with s
# avg energy -1.8619091260966236
# Opt_Points [-0.28950500394572903]  
# noisy energy -1.127 
# mitigated energy 
    
# without s
# avg energy -1.894064319269586
# Opt_Points [-0.2704871037925166]
# noisy energy 
# mitigated: -1.8709    


# 3 times 
# ideal = -1.860848342649004
# opt point(ideal): -0.33656965405055006
# noisy: -1.349447559967852



























