import numpy as np
from qiskit import Aer, QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import *
from qiskit.circuit import Parameter, parameter
from qiskit.quantum_info.operators import Operator
import copy


class SMOVQS():

    def __init__(self, n, reps = 1, backend_name=None):

        self.n = int(n)


        self.V = []
        self.pos = []
        self.np = reps * self.n + self.n
        self.reps = reps
        self.p = ParameterVector('θ', length=self.np)

        if backend_name is None:

            self.backend = Aer.get_backend('aer_simulator')


        else:

            assert type(backend_name) is str, 'backend_name must be string'

            try:
                device_backend = eval(backend_name+"()")
                self.backend = AerSimulator.from_backend(device_backend)
            except:
                print("backend_name didn't match any backend")
                print("Therefore, aer_simulator is choosen")
                self.backend = Aer.get_backend('aer_simulator')

    def twolocal(self):

        G = QuantumCircuit(self.n+1, name='psi')

        count = 0

        G.h([n+1 for n in range(self.n)])

        for i in range(self.reps):

            G.barrier()
            for n in range(self.n):
                G.cry(self.p[count], 0, n+1)
                count+=1

            G.barrier()
            for n in range(self.n):
                G.ccx(0,n+1, (n+1)%(self.n)+1)

        G.barrier()
        for n in range(self.n):
            G.cry(self.p[count], 0, n+1)
            count+=1
        G.barrier()
    


        

        self.G = G

        return self.G

    
    def return_state(self):

        G = QuantumCircuit(self.n, name='psi')

        count = 0
        G.h([n for n in range(self.n)])
        
        for i in range(self.reps):
            G.barrier()

            for n in range(self.n):
                G.ry(self.p[count], n)
                count+=1

            G.barrier()              
            for n in range(self.n):
                G.cx(n, (n+1)%(self.n))
            G.barrier()
            
        for n in range(self.n):
            G.ry(self.p[count], n)
            count+=1

        assert self.n % 2 == 0


        
        self.state = G

        return G

    def get_samples(self, theta, n_samples = 1000, method='ic'):

        """
        calculate inner product

        method : method to calculate inner product. method='ic' calculate inner product with interference circuit.
        """

        Q = QuantumCircuit(self.n+1)
        Q.h(0)
        Q = Q.compose(self.twolocal(), [i for i in range(self.n+1)])
        Q.h(0)

        Q.measure_all()

        params = {self.p[i] : theta[i] for i in range(self.np)}

        Q = Q.bind_parameters(params)
        result = execute(Q, self.backend, shots=n_samples).result()
        result = result.get_counts(0)

        prob = np.zeros(len(result))
        state_array = np.zeros((len(result),self.n+1), np.int8)

        for n, (key, item) in enumerate(result.items()):

            for i,j in enumerate(key):
                state_array[n,-(i+1)] = -((int(j) * 2) - 1)

            
            prob[n] = item / n_samples
        
        self.Q_inner = Q

        return state_array, prob

    def set_ham(self, ham_dict):

        '''

        ham_dict['ham'] is list of {'zz', 'z'}

        ham_dict['pos'] is list of {[i,j] , [i]} which represent sites on which local hamiltonian act

        '''
        assert set(ham_dict.keys()) == set(['c','ham','pos']), 'contains wrong keys / missing some keys'


        for key, ele in ham_dict.items():
            assert len(ele) == len(ham_dict['ham']), "length of each elements doesn't match"

        self.o_ham_dict = copy.copy(ham_dict)

        self.ham_dict = {
            'ham' : [],
            'pos' : [],
            'c' : []
        }


    
        i = 0
        for h, pos, c in zip(ham_dict['ham'], ham_dict['pos'], ham_dict['c']):

            n = len(h)

            ham_list = ['x','y','z']
            # h_ = set(h)
            for h_ in set(h):
                assert h_ in ham_list,'invalid character in "ham" '
            

            lh = QuantumCircuit(n+1, name='ham{}'.format(i))

            for i, h_ in enumerate(h):
                
                if h_ == 'x':
                    lh.cx(0,i+1)
                elif h_ == 'y':
                    lh.cy(0,i+1)
                elif h_ == 'z':

                    lh.cz(0,i+1)
            
            self.ham_dict['ham'].append(lh)
            self.ham_dict['pos'].append([0] + [j+1 for j in pos])
            self.ham_dict['c'].append(c)


    def b_factor(self, X):

        Pos = self.o_ham_dict['pos']
        C = self.o_ham_dict['c']
        energy = np.zeros(X.shape[0])
        # e = 0


        for i in range(X.shape[0]):
            x = X[i]
            x_ = x[1:]
            e = 0
            for pos, c in zip(Pos, C):
                lx = x_[pos]
                e += c * lx.prod()
            energy[i] = e
        
        return np.exp(-energy) * X[:,0]

    def energy(self, X):

        Pos = self.o_ham_dict['pos']
        C = self.o_ham_dict['c']
        energy = np.zeros(X.shape[0])
        # e = 0


        for i in range(X.shape[0]):
            x = X[i]
            x_ = x[1:]
            e = 0
            for pos, c in zip(Pos, C):
                lx = x_[pos]
                e += c * lx.prod()
            energy[i] = e
        
        return energy



        

    def return_ham(self, return_list=False):


        i = 0
        ham = np.zeros((2**self.n,2**self.n), dtype=np.complex128)
        ham_list_ = []
        ham_list = ['x','y','z']


        for h, pos, c in zip(self.o_ham_dict['ham'], self.o_ham_dict['pos'], self.o_ham_dict['c']):

            qc = QuantumCircuit(self.n)

            n = len(h)

            # h_ = set(h)
            for h_ in set(h):
                assert h_ in ham_list,'invalid character in "ham" '
            


            for i, h_ in enumerate(h):
                
                if h_ == 'x':
                    qc.x(pos[i])
                elif h_ == 'y':
                    qc.y(pos[i])
                elif h_ == 'z':
                    qc.z(pos[i])

            
            ham += c*Operator(qc).data
            ham_list_.append(Operator(qc).data)

        if return_list:
            return ham, ham_list_, 


        return ham

def sequential_opt(vqs, theta_0, n_iter = 10**2, n_samples = 10**4):

    index = np.random.randint(0, vqs.np, size = n_iter-1)

    theta = np.zeros((n_iter, vqs.np))
    cost = np.zeros(n_iter)
    theta[0] = theta_0

    cost[0] = return_cost(vqs, theta_0, n_samples)

    for j, i in enumerate(index):

        theta[j+1], cost[j+1] = optimize_smo(vqs, i, theta[j], n_samples, cost[j])

        # if cost[j+1] < cost[j]:
        #     theta[j+1] = theta[j]
        #     cost[j+1] = cost[j]



    return theta, cost



    
def optimize_smo( vqs, i, theta, n_samples, cost_0 = False):

    assert 0 <= i < theta.shape[0]
    
    theta0 = theta[i]
    
    Z = np.array([-np.pi, 0, np.pi])

    cost = []

    if cost_0:
        for z in Z:
            if z==0:
                cost.append(cost_0)
                continue
            theta_ = np.copy(theta )
            theta_[i] += z
            cost.append(return_cost(vqs, theta_, n_samples))
    
    else:
        for z in Z:
            theta_ = np.copy(theta )
            theta_[i] += z
            cost.append(return_cost(vqs, theta_, n_samples))


    gamma = (cost[0] - cost[2]) / (2*cost[1] - cost[0] - cost[2])
    theta_hat = theta0 - 2*np.arctan(gamma)
    theta_[i] = theta_hat

    return theta_ , return_cost(vqs, theta_, n_samples)


def return_cost(vqs, theta, n_samples):

    X, proba = vqs.get_samples_ic(theta, n_samples = n_samples)
    return (vqs.b_factor(X) * proba ).sum()