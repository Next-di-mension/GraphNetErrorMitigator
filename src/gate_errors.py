# fake backend
from qiskit.providers.fake_provider import FakeManila, FakeMelbourne, FakeGuadalupe
import numpy as np 

NUM_NODES = 14

def gate_error_matrix(backend_fake):
    m = np.zeros((NUM_NODES,NUM_NODES))
    j = -1
    for i in range(len(backend_fake.properties().to_dict()['gates'])):

        if backend_fake.properties().to_dict()['gates'][i]['gate'] == 'x':
            j = j + 1
            m[j,j] = backend_fake.properties().to_dict()['gates'][i]['parameters'][0]['value']
            if j == NUM_NODES-1:
                break

    j = -1
    for i in range(len(backend_fake.properties().to_dict()['gates'])):
        if backend_fake.properties().to_dict()['gates'][i]['gate'] == 'id':
            j = j + 1
            m[j,j] = m[j,j] +  backend_fake.properties().to_dict()['gates'][i]['parameters'][0]['value']
            if j == NUM_NODES-1:
                break
    j = -1
    for i in range(len(backend_fake.properties().to_dict()['gates'])):
        if backend_fake.properties().to_dict()['gates'][i]['gate'] == 'sx':
            j = j + 1
            m[j,j] = m[j,j] +  backend_fake.properties().to_dict()['gates'][i]['parameters'][0]['value']
            if j == NUM_NODES-1:
                break
    return m
l = 0   
k = 0
# two qubit gate error matrix
def two_qubit_gate_error_matrix(backend_fake):
    n = np.zeros((NUM_NODES,NUM_NODES))

    for i in range(len(backend_fake.properties().to_dict()['gates'])):

        if backend_fake.properties().to_dict()['gates'][i]['gate'] == 'cx':
            l = backend_fake.properties().to_dict()['gates'][i]['qubits'][0]
            k = backend_fake.properties().to_dict()['gates'][i]['qubits'][1]
            if l < NUM_NODES and k < NUM_NODES:
                n[l,k] = backend_fake.properties().to_dict()['gates'][i]['parameters'][0]['value']

    return n

# backend_fake = FakeGuadalupe()
backend_fake = FakeMelbourne()

one_qubit_mat= gate_error_matrix(backend_fake)
two_qubit_mat = two_qubit_gate_error_matrix(backend_fake)
node_embd_mat = one_qubit_mat + two_qubit_mat



