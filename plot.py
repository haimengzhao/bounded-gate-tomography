import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from qiskit.quantum_info import random_clifford, random_unitary
from tqdm import tqdm
import tensorcircuit as tc
K = tc.set_backend("jax")

import scienceplots
plt.style.use(['science','no-latex'])

n = 10000
focus_n = 4
gate_set_size = 2
sample_size_max = 70
n_run = 20000
G_max = 10
G_list = range(1, G_max+1)
sample_size_list = np.arange(1, sample_size_max+1)



if focus_n is not None:
    fid_list = np.load(f"n={n}_G={G_max}_nrun={n_run}_focus.npy")
else:
    fid_list = np.load(f"n={n}_G={G_max}_nrun={n_run}.npy")

fid_mean = jnp.mean(fid_list, axis=0)
fid_med = jnp.quantile(fid_list, 0.5, axis=0)
fid_std = fid_list.std(axis=0)

def median_of_mean(x, k):
    shape = x.shape
    return jnp.median(x.reshape(k, -1, *shape[1:]).mean(axis=0), axis=0)

fid_med_mean = median_of_mean(fid_list, 10)

plt.figure(figsize=(8, 6))
thres_list = [0.7, 0.8, 0.9, 0.95, 0.99]
for thres in thres_list:
    ind = np.array(np.argmax(fid_mean>thres, axis=-1))
    plt.plot(G_list, sample_size_list[ind], linestyle='-', marker='o', label=f"$F={thres}$")
ind = np.array(np.argmax(fid_med>0.999, axis=-1))
plt.plot(G_list, sample_size_list[ind], linestyle='--', marker='o', label="$F_{med}=0.999$")
plt.contourf(G_list, sample_size_list, fid_mean.T, levels=100)
plt.colorbar()
plt.xlabel("Gate Size $G$")
# plt.xscale("log")
plt.ylabel("Sample Size $N$")
plt.title(f"$n={n}$, {n_run} repetitions")
plt.legend()

if focus_n is not None:
    plt.savefig(f"n={n}_G={G_max}_nrun={n_run}_focus.pdf")
else:   
    plt.savefig(f"n={n}_G={G_max}_nrun={n_run}.pdf")

plt.show()