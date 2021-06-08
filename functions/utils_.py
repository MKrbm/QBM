import numpy as np
from qiskit import *
from qiskit.circuit import Parameter, parameter


class estimate_params():

    def __init__(self, Vs, Vs_qargs, gs, gs_qargs, ham_dict , n, params=None):
        
        self.Vs = Vs
        self.Vs_qargs = Vs_qargs
        self.gs = gs
        self.gs_qargs = gs_qargs
        self.ham_dict = ham_dict
        self.n = n
        self.params = params 

    def estimate_Fisher(self, der_pos = [0,1], NUM=10000):

        assert self.params, 'please provide params'

        N = len(self.Vs)
        L = 1


        q = QuantumRegister(self.n+1)
        c = ClassicalRegister(1)
        A = QuantumCircuit(q,c, name='Fisher')
        A.h(0)


        for i in range(N):
            if i in der_pos:
                A.x(0)
                A.append(self.gs[i], [q[k] for k in self.gs_qargs[i]])
            A.append(self.Vs[i],[q[k] for k in self.Vs_qargs[i]])
        A.h(0)
        A.measure(0, 0)

        A = A.bind_parameters(self.params)

        backend = BasicAer.get_backend('qasm_simulator')

        if NUM > 10000:
            L = NUM//10000
            NUM = 10000
        
        A_results = []

        for _ in range(L):
            
            backend = BasicAer.get_backend('qasm_simulator')
            result = execute(A, backend, shots=NUM).result()
            counts  = result.get_counts(A)
            if '0' in counts.keys():
                A_results.append((counts['0']/NUM)*2 -1)
            else:
                A_results.append(-1)

        return np.mean(A_results)


    def estimate_C(self ,der_pos = [1] ,NUM = 10000):

        N = len(self.Vs)
        M = len(self.ham_dict['ham'])
        L = 1


        C_hat = 0

        q = QuantumRegister(self.n+1)
        c = ClassicalRegister(1)



        for m in range(M):


            C = QuantumCircuit(q,c, name='Fisher')
            C.h(0)
            C.rz(np.pi/2,0)
            


            for i in range(N):
                if i in der_pos:
                    C.x(0)
                    C.append(self.gs[i], [q[k] for k in self.gs_qargs[i]])
                C.append(self.Vs[i],[q[k] for k in self.Vs_qargs[i]])
            C.x(0)
            C.append(
                self.ham_dict['ham'][m],
                [q[k] for k in self.ham_dict['pos'][m]]
                )
            C.h(0)
            C.measure(0, 0)

            C = C.bind_parameters(self.params)



            backend = BasicAer.get_backend('qasm_simulator')

            if NUM > 10000:
                L = NUM//10000
                NUM = 10000
            
            C_results = []

            for _ in range(L):
                
                backend = BasicAer.get_backend('qasm_simulator')
                result = execute(C, backend, shots=NUM).result()
                counts  = result.get_counts(C)
                if '0' in counts.keys():
                    C_results.append((counts['0']/NUM)*2 -1)
                else:
                    C_results.append(-1)
                
            
            C_hat += np.mean(C_results) * (-1) * self.ham_dict['c'][m] 


        return C_hat


    def estimate_AC(self, NUM = 10000):

        N = len(self.Vs)

        C_array = np.zeros(N)
        A_array = np.eye(N)

        for i in range(N):
            for j in range(i):
                A_array[i,j] = \
                    self.estimate_Fisher(der_pos = [i, j] ,NUM = NUM)
                A_array[j, i] = A_array[i,j]

        for i in range(N):
            C_array[i] = \
                self.estimate_C(der_pos = [i] ,NUM = NUM)
        
        return A_array/4, C_array/2



class VQS():

    def __init__(self, n):

        self.q = QuantumRegister(int(n))
        self.n_params = int(2*n)
        self.n = int(n)

        self.P = [Parameter('θ{}'.format(i)) for i in range(self.n_params)]

        self.V = []
        self.pos = []

        for i in range(self.n):
            
            if (i+1)%self.n == 0:
                v = QuantumCircuit(self.n, name='V{}'.format(i))
                v.ry(self.P[i], self.n-1)
                for j in range(self.n):
                    v.cnot((self.n-1+j)%self.n, j%self.n)
                self.V.append(v)
                self.pos.append([i for i in range(self.n)])

            else:
                v = QuantumCircuit(1, name='V{}'.format(i))
                v.ry(self.P[i], 0)
                self.V.append(v)
                self.pos.append([i])
            
        for i in range(self.n):
            
            i_ = i + self.n

            if (i+1)%self.n == 0:
                v = QuantumCircuit(self.n, name='V{}'.format(i_))
                v.ry(self.P[i_], self.n-1)
                for j in range(int(self.n/2)):
                    v.cnot(j, j + int(self.n/2))
                self.V.append(v)
                self.pos.append([i for i in range(self.n)])

            else:
                v = QuantumCircuit(1, name='V{}'.format(i_))
                if i < int(self.n/2):
                    v.ry(self.P[i_] + np.pi/2, 0)
                else:
                    v.ry(self.P[i_], 0)
                self.V.append(v)
                self.pos.append([i])


        # for i in range(self.n):

        #     v = QuantumCircuit(1, name='V{}'.format(i+2*self.n))
        #     v.ry(self.P[i+2*self.n], 0)
        #     self.V.append(v)
        #     self.pos.append([i])

        


        self.gs = []
        self.g_pos = []


        # q[self.n] is ancilla bit for computing A and C

        for i in range(self.n_params):
            g = QuantumCircuit(2, name='g{}'.format(i))
            g.cy(0,1)
            self.gs.append(g)
            self.g_pos.append([0, i%self.n+1])

        self.pos_est = [[j+1 for j in pos] for pos in self.pos]


    def set_ham(self, ham_dict):

        '''

        ham_dict['ham'] is list of {'zz', 'z'}

        ham_dict['pos'] is list of {[i,j] , [i]} which represent sites on which local hamiltonian act

        '''

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


            # if h == 'z':
            #     lh = QuantumCircuit(2, name='ham{}'.format(i))
            #     lh.cz(0,1)
            #     self.ham_dict['ham'].append(lh)
            #     self.ham_dict['pos'].append([0] + [j+1 for j in pos])
            #     self.ham_dict['c'].append(c)

            # elif h == 'x':
            #     lh = QuantumCircuit(2, name='ham{}'.format(i))
            #     lh.cx(0,1)
            #     self.ham_dict['ham'].append(lh)
            #     self.ham_dict['pos'].append([0] + [j+1 for j in pos])
            #     self.ham_dict['c'].append(c)

            # elif h == 'zz':
            #     lh = QuantumCircuit(3, name='ham{}'.format(i))
            #     lh.cz(0,1)
            #     lh.cz(0,2)
            #     self.ham_dict['ham'].append(lh)
            #     self.ham_dict['pos'].append([0] + [j+1 for j in pos])
            #     self.ham_dict['c'].append(c)

            # elif h == 'xx':
            #     lh = QuantumCircuit(3, name='ham{}'.format(i))
            #     lh.cx(0,1)
            #     lh.cx(0,2)
            #     self.ham_dict['ham'].append(lh)
            #     self.ham_dict['pos'].append([0] + [j+1 for j in pos])
            #     self.ham_dict['c'].append(c)
            # i+=1


    def state(self):

        rho = QuantumCircuit(self.q, name='rho')

        for i in range(self.n_params):
            rho.append(self.V[i], [self.q[j] for j in self.pos[i]])

        # assert n%2 == 0, 'n must be even number'

        return rho

        # n_params = n * 2


class VQS2(VQS):

    def __init__(self, n):

        self.q = QuantumRegister(int(n))
        self.n_params = int(n)
        self.n = int(n)

        self.P = [Parameter('θ{}'.format(i)) for i in range(self.n_params)]

        self.V = []
        self.pos = []
        self.n_prime = int(self.n/2)

        for i in range(self.n_prime):
            
            if (i+1)%self.n_prime == 0:
                v = QuantumCircuit(self.n_prime, name='V{}'.format(i))
                v.ry(self.P[i], self.n_prime-1)
                for j in range(self.n_prime):
                    v.cnot((self.n_prime-1+j)%self.n_prime, j%self.n_prime)
                self.V.append(v)
                self.pos.append([i for i in range(self.n_prime)])

            else:
                v = QuantumCircuit(1, name='V{}'.format(i))
                v.ry(self.P[i], 0)
                self.V.append(v)
                self.pos.append([i])
            
        for i in range(self.n_prime):
            
            i_ = i + self.n_prime

            if (i+1)%self.n_prime == 0:
                v = QuantumCircuit(self.n, name='V{}'.format(i_))
                v.ry(self.P[i_] + np.pi/2, self.n_prime-1)
                for j in range(int(self.n/2)):
                    v.cnot(j, j + int(self.n/2))
                self.V.append(v)
                self.pos.append([i for i in range(self.n)])


            else:
                v = QuantumCircuit(1, name='V{}'.format(i_))
                if i < int(self.n_prime/2):
                    v.ry(self.P[i_] + np.pi/2, 0)
                else:
                    v.ry(self.P[i_], 0)
                self.V.append(v)
                self.pos.append([i])



        self.gs = []
        self.g_pos = []


        # q[self.n] is ancilla bit for computing A and C

        for i in range(self.n_params):
            g = QuantumCircuit(2, name='g{}'.format(i))
            g.cy(0,1)
            self.gs.append(g)
            self.g_pos.append([0, i%self.n_prime+1])

        self.pos_est = [[j+1 for j in pos] for pos in self.pos]






