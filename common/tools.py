from scipy.integrate import solve_ivp
from functools import partial

import numpy as np
import warnings

from sbi import utils as utils
from sbi.inference.base import infer

import warnings
import numpy as np
from pyfmi import load_fmu

# Reactor constants
V = 150  # L
V_c =10 #    L
Q=100# L/min

# Reaction constants
E_a  = 83140#   J/mol 
R  = 8.314 #    J/K/mol 
H_r  =-2E5 #       cal/mol 

# Fluid constants
C_p = 1 # cal/g/k
C_pc  =1 #   cal/g/k 
rho = 1000 # g/L
rho_c = 1000 #g/l


def cstr_model(t,y,params,inlet_conditions):
    """
    Differential equations of CSTR model 

    Args:
        t(float): Time
        t(List): Variables C, T and T_c
        params(List): Parameters UA, k_0
        inlet_conditions(List): Inlet conditions for simulator (C_i, T_i, T_ci, Q)
    """
    
    C_i, T_i, T_ci,Q_c = inlet_conditions
    UA,k_0 = params
    C, T, T_c = y

    k = k_0*np.exp(-E_a/(R*T))
    
    dC_dt = (Q/V)*(C_i-C) - k*C
    dT_dt = (Q/V)*(T_i-T) - H_r*k*C/(rho*C_p) - UA*(T-T_c)/(rho*C_p*V)
    dTc_dt = (Q_c/V_c)*(T_ci - T_c) + UA * (T-T_c)/(rho_c*C_pc*V_c)
    
    return dC_dt, dT_dt, dTc_dt

def scipy_simulator(inlet_conditions, params, sim_time = 20):
    """
    Simulator wrapper for CSTR model using SciPy
    
    args:
        inlet_conditions(List): Inlet conditions for simulator (C_i, T_i, T_ci, Q)
        params(List): Parameters in the order of UA, k_0
        sim_time(float): Simulation time in minutes
    """
    
    y0 = [0.1, 430, 416]
        
    res = solve_ivp(cstr_model,(0,sim_time),y0, method='RK45',args=(params,inlet_conditions))
    
    sol = res.y
    output = sol[:,-1]
    
    # Check tolerance 
    if not meets_convergence(output, sol[:,-5], y0):
        return [np.nan, np.nan, np.nan]
    
    return output


def run_sbi(inlet_conditions, observation,num_simulations=100):
    """
    SBI wrapper that initialises the simulator with initial conditions and runs SBI on it. 

    Args:
        inlet_conditions(List): Inlet conditions for simulator (C_i, T_i, T_ci, Q)
        observation(List): Variables C, T and T_c to use for the inference of parameters
        num_simulations(int): Number of simulations to perform for SBI
    """
    simulator = partial(pre_simulator, inlet_conditions)
    prior = utils.BoxUniform(low=[6E5,1E10], 
                             high=[10E5,10E10])

    posterior = infer(simulator, prior, method='SNPE', num_simulations=num_simulations, num_workers=5)
    samples = posterior.sample((10000,), x=observation,show_progress_bars=False)
    return samples
    

# Define simulator wrapper
def fmu_simulator(inlet_conditions, params, model, opts=None, sim_time =20):


    Ci, Ti, Tci, Qc = inlet_conditions
    UA, k_0 = params
    y0 = np.array([0.1,430,416])
    model.reset()

    start_values={'UA':UA,'k_0':k_0,'Ci':Ci,'Ti':Ti,'Tci':Tci,'Qc':Qc}
    
    for item in start_values:
        model.set(item, start_values[item])

    result = model.simulate(start_time=0, final_time=sim_time,options=opts)

    output = np.array([result['C'][-1], result['T'][-1], result['Tc'][-1]])
    output_5 = np.array([result['C'][-5], result['T'][-5], result['Tc'][-5]])

    if not meets_convergence(output, output_5, y0):
        raise RuntimeError('Convergence not met..')

    return output

def meets_convergence(output, output_5, scaling):
    # Check tolerance 
    abs_error = abs(output - output_5)
    rel_error = abs_error*100/scaling
    error_percentage = 1

    # If simulation is outside of local tolerance, then issue warning and return array of null values
    if any(rel_error>error_percentage):
        warnings.warn(f'Solution not converged. {rel_error}', RuntimeWarning)
        return False
    
    return True


if __name__== "__main__":
    inlet_conditions = [0.97, 351.5, 351.6, 150]
    true_params = [7E5, 7.2E10]
    
    # Use simulator to get observation
    observation = pre_simulator(inlet_conditions, true_params)
    print(observation)

    # Run SBI to get the mean parameters for that observation
    samples = run_sbi(inlet_conditions, observation)
    sbi_param_1 = np.mean([float(sample[0]) for sample in samples])
    sbi_param_2 = np.mean([float(sample[1]) for sample in samples])
    print([sbi_param_1, sbi_param_2])
