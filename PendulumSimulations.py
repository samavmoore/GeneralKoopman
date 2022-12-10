import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import math
import random as rand
from scipy.stats import truncnorm

# define pendulum dynamics function 
def pend(t, thetas, g, l):
    theta, theta_dot = thetas

    theta_ddot = -g/l*math.sin(theta)
    return [theta_dot, theta_ddot]

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def get_pendulum_data(n_ICs, n_configs, start_time, stop_time, traj_len):
    configs = np.zeros([n_configs, 2])
    data = np.zeros([n_configs*n_ICs*traj_len, 4])
    time = np.linspace(start_time, stop_time, traj_len)

    theta_dist = get_truncated_normal(mean=0, sd=2, low=-3.1, upp=3.1)

    for i in range(n_configs):
            # get random parameters for g and l 
            g = rand.uniform(1, 12)
            l = rand.uniform(.1, 1)
            configs[i,:] = [g, l]

            # repeat and stack parameters for each trajectory in this sub-experiment
            repeated_params = np.tile([g,l], (n_ICs*traj_len,1))

            # get the starting and ending indices for each parametric configuration
            exp_starting_index = i*n_ICs*traj_len
            exp_ending_index = (i+1)*n_ICs*traj_len

            # assign parameters to data
            data[exp_starting_index:exp_ending_index, 2:] = repeated_params

            for traj in range(n_ICs):

                # get starting and ending indices for each trajectory
                traj_starting_index = exp_starting_index + traj*traj_len
                traj_ending_index = exp_starting_index + (traj+1)*traj_len

                # get random initial conditions
                
                theta_0 = theta_dist.rvs()
                safety = g/l-.02
                upperbound = math.sqrt(2*safety)*math.sqrt(1+math.cos(theta_0))
                lowerbound = -upperbound
                theta_dot_0 = rand.uniform(lowerbound, upperbound)

                # simulate dynamics
                sol = solve_ivp(pend, t_span=[start_time, stop_time], y0=[theta_0, theta_dot_0], args=(g, l), t_eval=time)

                # get trajectory
                z = sol.y
                trajectory = z.T

                # assign trajectory to data
                data[traj_starting_index:traj_ending_index, 0:2] = trajectory

    return configs, data


if __name__=="__main__":

########----------- Context Pre-train Data ---------------------

    n_context_configs = 1000

    context = np.zeros([1000, 2])

    for i in range(n_context_configs):
        g = rand.uniform(1, 12)
        l = rand.uniform(.2, 1)
        context[i, :] = [g, l]

##########---------------- Main Training Data -----------------
    # initialize simulation parameters
    traj_len = 51
    start_time = 0
    stop_time = 1
    n_train_configs = 50
    n_train_ICs = 100
    
    train_configs, train_data = get_pendulum_data(n_ICs=n_train_ICs, n_configs=n_train_configs, traj_len=traj_len, start_time=0, stop_time=3)

    np.save("Pendulum_train_data", train_data)
    np.save("Pendulum_train_configs", train_configs)

########------------ Main Test/Validation Data ----------------
    n_val_configs = 10
    n_val_ICs = 2

    val_configs, val_data = get_pendulum_data(n_ICs=n_val_ICs, n_configs=n_val_configs, traj_len=traj_len, start_time=0, stop_time=3)

    np.save("Pendulum_val_data", val_data)
    np.save("Pendulum_val_configs", val_configs)
