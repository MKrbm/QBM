from . import utils_ 
import numpy as np
from sklearn.linear_model import Ridge
from qiskit import *
import qiskit.quantum_info as qi
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
import scipy
import copy

class time_evolution():

    def __init__(self, VQS, theta=None, backend_name=None):

        self.iter = 0

        if theta is None:
            self.theta = np.zeros(VQS.n_params)
        else:
            self.theta = theta
        

        self.backend_name = backend_name
        

        self.estimator = utils_.estimate_params3(VQS, self.backend_name)
        self.VQS = VQS

    def setVQS(self, VQS):
        self.estimator = utils_.estimate_params3(VQS, self.backend_name)
        self.VQS = VQS
    
    def reset(self):
        self.theta = np.zeros(self.VQS.n_params)
        self.iter = 0
    
    def seq_QITE(self,ham_dict, rate = 10, p_num =  None, n_iter=10**4):

        ham_dict['c'] = np.array(ham_dict['c'])

        if self.iter == 0:

            N = np.int(np.abs(ham_dict['c']).max() * rate)
            if p_num:
                N = p_num

            self.VQS.set_ham(ham_dict)
            self.estimator = utils_.estimate_params3(self.VQS, self.backend_name)
            gibbs = scipy.linalg.expm(-(self.VQS.return_ham()))
            gibbs /= np.trace(gibbs)
            gibbs = DensityMatrix(gibbs)

            self.iter += 1
            self.old_c = np.copy(ham_dict['c'])
            return self.QITE(N = N, n_iter = n_iter, gibbs = gibbs)
        
        if self.iter > 0:

            ham_dict_prime = copy.copy(ham_dict)
            ham_dict_prime['c'] = ham_dict['c'] - self.old_c

            print(ham_dict_prime['c'])

            N = max(np.int(np.abs(ham_dict_prime['c']).max() * rate), 1)
            if p_num:
                N = p_num
            self.VQS.set_ham(ham_dict_prime)
            self.estimator = utils_.estimate_params3(self.VQS, self.backend_name)

            VQS = utils_.VQS(self.VQS.n)
            VQS.set_ham(ham_dict)
            gibbs = scipy.linalg.expm(-(VQS.return_ham()))
            gibbs /= np.trace(gibbs)
            gibbs = DensityMatrix(gibbs)

            self.iter += 1
            self.old_c = np.copy(ham_dict['c'])
            return self.QITE(N = N, n_iter = n_iter, gibbs = gibbs)

    
    def QITE(self, N = 10, n_iter = 10**3, gibbs=None):
        
        delta = (1/2)/N
        tau = 0
        fid_list = []
        rho_list = []
    #     theta = np.zeros(VQS_.n_params)
        i = 0
        while True:

            self.estimator.set_theta(self.theta)
            A,C = self.estimator.estimate_AC(NUM = n_iter)
            
            clf = Ridge(fit_intercept=False, alpha=1e-6)
            clf.fit(A, C)
    #         clf = LinearRegression().fit(A, C)
            self.theta += clf.coef_ * delta
            tau += delta
            

            
            params = {self.VQS.P[i] : self.theta[i] for i in range(self.VQS.n_params)}
            rho1 = self.VQS.state().bind_parameters(params)
            out = Statevector.from_instruction(rho1)
            rho1_ = partial_trace(out,[l for l in range(int(self.VQS.n/2), self.VQS.n)])
            if gibbs:
                Fidelity = qi.state_fidelity(rho1_, gibbs)
                fid_list.append(Fidelity)
            rho_list.append(rho1_)
            

            
            
            if round(tau,3) >= 1/2:
                break
            i += 1
        
        if gibbs:
            return rho_list, fid_list

        else:
            return rho_list
