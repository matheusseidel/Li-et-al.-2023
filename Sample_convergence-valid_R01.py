# ----------------------------------------------------------- #
#                            DMD                              #
# ----------------------------------------------------------- #

# Author:               Matheus Seidel (matheus.seidel@coc.ufrj.br)
# Revision:             04
# Last update:          11/07/2023

# Description:
'''  
This code performs the Dynamic Mode Decomposition on fluid simulation data. It uses the PyDMD library to  to generate the DMD modes
and reconstruct the approximation of the original simulation.
The mesh is read by meshio library using vtk files. The simulation data is in h5 format and is read using h5py.
Details about DMD can be found in:
Schmid, P. J., "Dynamic Mode Decomposition of Numerical and Experimental Data". JFM, Vol. 656, Aug. 2010,pp. 5–28. 
doi:10.1017/S0022112010001217
'''

# Last update
'''
Replicate Li et al.
'''

# ----------------------------------------------------------- #

import matplotlib.pyplot as plt
import h5py
import meshio
import numpy as np
from scipy import linalg
import time

# ------------------- Parameter inputs ---------------------- #

start_time = time.time()

#r = 100                # SVD rank
dt = 1e-5              # time step value

# ----------------------- Function--------------------------- #

def read_h5_libmesh(filename, dataset):
    """
    Function used to read nodal values from H5 files.

    Parameters
    ----------
    filename : str
        String containing the filename for reading files in h5 (libMesh and EdgeCFD).
    dataset : str
        String containing the dataset desired.

    Returns
    -------
    array : np.array
        Numpy array containing nodal values.
    """
    h5_file = h5py.File(filename, "r")
    data = h5_file[(dataset)]
    data_array = np.array(data, copy=True)
    h5_file.close()
    return data_array

# ----------------- Reading pressure data ------------------- #

n_max = 16               # Number of cycles
#r = 10
#ti = 1
#tf = 15001

ncycles = np.arange(1, n_max + 1)
grandmean = np.zeros(n_max)

#n = 23401                              # number of nodes
#m = tf - ti                            # number of timesteps

#snapshots_p = read_h5_libmesh('Midplane_xy_Vm_25_cycles_nointerp.mat', 'Vm_all')

#snapshots_p = np.transpose(snapshots_p)

#print(f'Pressure snapshots matrix: {snapshots_p.shape}')
#print()

Start_frame = 120080
Cycle_frame = np.array([121000, 121928, 122808, 123775, 124737, 125695, 126613, 127582, 128599, 129505, 130549, 131559, 132541, 133637, 134581, 135546, 136552, 137621, 138638, 139595, 140539, 141469, 142494, 143480])

#for N in range(0, n_max):
for N in range(12, 13):

    print()
    print(f'Running cycle {N + 1}')
    print()

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'w') as log:
        log.write('Log: ' + '\n')  
        log.write('\n')  

    End_frame = Cycle_frame[N]

    snapshots_p = read_h5_libmesh('C:/Users/Matheus Seidel/OneDrive/Doutorado/1_Data/Li et al. 2023/Convergence Test/Midplane_xy_Vm_25_cycles_nointerp.mat', 'Vm_all')

    snapshots_p = np.transpose(snapshots_p)

    print(f'Pressure snapshots matrix: {snapshots_p.shape}')
    print()

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        snapshots_p_str = str(snapshots_p)
        log.write('snapshots_p: '+ snapshots_p_str + '\n')
        shape_str = str(snapshots_p.shape)
        log.write(shape_str + '\n')
        log.write('\n') 

    ti = 0                  
    tf = End_frame - Start_frame + 1    

    Vm_all_mean = snapshots_p[:, ti:tf].mean(axis=1)
    Vm_all_f = snapshots_p[:, ti:tf]

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        Vm_all_mean_str = str(Vm_all_mean)
        log.write('Vm_all_mean: '+ Vm_all_mean_str + '\n')
        shape_str = str(Vm_all_mean.shape)
        log.write(shape_str + '\n')
        log.write('\n') 
        Vm_all_f_str = str(Vm_all_f)
        log.write('Vm_all_f (sem subtração): '+ Vm_all_f_str + '\n')
        shape_str = str(Vm_all_f.shape)
        log.write(shape_str + '\n')
        log.write('\n') 

    for i in range(0, tf):
        Vm_all_f[:, i] = Vm_all_f[:, i] - Vm_all_mean

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        Vm_all_f_str = str(Vm_all_f)
        log.write('Vm_all_f: '+ Vm_all_f_str + '\n')
        shape_str = str(Vm_all_f.shape)
        log.write(shape_str + '\n')
        log.write('\n') 

    print('Setting X1 and X2')
    print()

    X1 = Vm_all_f[:, ti:tf-1]
    X2 = Vm_all_f[:, ti+1:tf]

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        X1_str = str(X1)
        log.write('X1: '+ X1_str + '\n')
        shape_str = str(X1.shape)
        log.write(shape_str + '\n')
        log.write('\n')  
        X2_str = str(X2)
        log.write('X2: '+ X2_str + '\n')
        shape_str = str(X2.shape)
        log.write(shape_str + '\n')
        log.write('\n') 

    r = X1.shape[1]

    print(f'Rank {r}')
    print()

    print('Running SVD')
    print()

    U, S, Vt = linalg.svd(X1)

    V = np.transpose(Vt)

    r = min(r, U.shape[1])

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        r_str = str(r)
        log.write('r: '+ r_str + '\n')
        log.write('\n')

    print('Truncating SVD')
    print()

    U_r = U[:, 0:r]
    S_r = S[0:r]
    V_r = V[:, 0:r]

    SM = np.diag(S)
    SM_r = np.diag(S_r)

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        U_r_str = str(U_r)
        log.write('U_r: '+ U_r_str + '\n')
        shape_str = str(U_r.shape)
        log.write(shape_str + '\n')
        log.write('\n')  
        SM_r_str = str(SM_r)
        log.write('SM_r: '+ SM_r_str + '\n')
        shape_str = str(SM_r.shape)
        log.write(shape_str + '\n')
        log.write('\n')  
        V_r_str = str(V_r)
        log.write('V_r: '+ V_r_str + '\n')
        shape_str = str(V_r.shape)
        log.write(shape_str + '\n')
        log.write('\n') 

    print('Calculating Atilde')
    print()

    Atilde = np.dot(np.transpose(U_r), X2)
    Atilde = np.dot(Atilde, V_r)
    Atilde = np.dot(Atilde, np.linalg.inv(SM_r))

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        Atilde_str = str(Atilde)
        log.write('Atilde: '+ Atilde_str + '\n')
        shape_str = str(Atilde.shape)
        log.write(shape_str + '\n')
        log.write('\n') 

    print('Calculating eigs of Atilde')
    print()

    D, W_r = np.linalg.eig(Atilde)
    D, W_r = linalg.eig(Atilde)

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        D_str = str(D)
        log.write('D: '+ D_str + '\n')
        shape_str = str(D.shape)
        log.write(shape_str + '\n')
        log.write('\n') 
        W_r_str = str(W_r)
        log.write('W_r: '+ W_r_str + '\n')
        shape_str = str(W_r.shape)
        log.write(shape_str + '\n')
        log.write('\n')

    Phi_p = np.dot(U_r, W_r)

    omega = np.log(D)/dt

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        Phi_p_str = str(Phi_p)
        log.write('Phi_p: '+ Phi_p_str + '\n')
        shape_str = str(Phi_p.shape)
        log.write(shape_str + '\n')
        log.write('\n')
        omega_str = str(omega)
        log.write('omega: '+ omega_str + '\n')
        shape_str = str(omega.shape)
        log.write(shape_str + '\n')
        log.write('\n')

    x1 = X1[:, 0]

    alpha_p, resid, rank, s  = np.linalg.lstsq(Phi_p, x1, rcond=-1) # rcond=-1 or None

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        alpha_p_str = str(alpha_p)
        log.write('alpha_p: '+ alpha_p_str + '\n')
        shape_str = str(alpha_p.shape)
        log.write(shape_str + '\n')
        log.write('\n')

    rss = alpha_p.shape[0]

    tr = X1.shape[1]

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        rss_str = str(rss)
        log.write('rss: '+ rss_str + '\n')
        log.write('\n')
        tr_str = str(tr)
        log.write('tr: '+ tr_str + '\n')
        log.write('\n')

    time_dynamics = np.zeros((rss, tr), dtype = 'complex_')

    t = dt*np.arange(0, tr)

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        t_str = str(t)
        log.write('t: '+ t_str + '\n')
        shape_str = str(t.shape)
        log.write(shape_str + '\n')
        log.write('\n')

    print('Updating time dynamics')
    print()

    for iter in range(0, tr):
        time_dynamics[:, iter] = (alpha_p * np.exp(omega * t[iter]))

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        time_dynamics_str = str(time_dynamics)
        log.write('time_dynamics: '+ time_dynamics_str + '\n')
        shape_str = str(time_dynamics.shape)
        log.write(shape_str + '\n')
        log.write('\n')

    Xdmd = np.dot(Phi_p, time_dynamics)

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        Xdmd_str = str(Xdmd)
        log.write('Xdmd: '+ Xdmd_str + '\n')
        shape_str = str(Xdmd.shape)
        log.write(shape_str + '\n')
        log.write('\n')

    eDMD = 0.1 * abs((X1 - Xdmd) / (X1 + 1e-20))

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        eDMD_str = str(eDMD)
        log.write('eDMD: '+ eDMD_str + '\n')
        log.write('\n')

    eDMD_mean = np.zeros(X1.shape[1])

    print('Calculating eDMD_mean')
    print()

    for i in range(0, X1.shape[1]):
        eDMD_mean[i] = np.mean(eDMD[:, i])

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        eDMD_mean_str = str(eDMD_mean)
        log.write('eDMD_mean: '+ eDMD_mean_str + '\n')
        log.write('\n')

    grandmean[N] = np.mean(eDMD_mean)

    with open(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Log_N_{N + 1}.txt', 'a') as log:
        grandmean_str = str(grandmean[N])
        log.write('grandmean: '+ grandmean_str + '\n')
        log.write('\n') 

    print(f'{grandmean[N]}')

print(ncycles)
print()
print(grandmean)

plt.plot(ncycles, grandmean, color="blue")
plt.xlabel('ncycles')
plt.ylabel('Error')
plt.title('Error x ncyles')
plt.savefig(f'C:/Users/Matheus Seidel/OneDrive/Doutorado/3_Outputs/Li et al. 2023/Validation/Error x ncyles - P_{r}_{n_max}_{ti}_{tf}')
plt.show()

end_time = time.time()

elapsed_time = end_time - start_time
print("Elapsed time: ", elapsed_time)
