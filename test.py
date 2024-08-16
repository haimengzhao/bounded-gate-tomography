# set visible device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
# os.environ["JAX_PLATFORMS"]="cpu"

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

from circuit import enumerate_gates, random_arch, random_gates, fidelity_batch_candidates, get_relevant_qubits, focus_on_relevant_qubits
from learn import search_batch_G
from shadow import get_rotation

n = 10000
focus_n = None
shallow = True
gate_set_size = 2
sample_size_max = 60
n_run = 25000
G_max = 6
G_list = range(1, G_max+1)
sample_size_list = np.arange(1, sample_size_max+1)

candidates_ind_cache = {G: enumerate_gates(gate_set_size, G) for G in G_list}

fid_list = np.ones(shape=(n_run, len(G_list), len(sample_size_list)))

key = jax.random.PRNGKey(0)

for run in tqdm(range(n_run)):
    
    gate_set = jnp.array([random_unitary(4).to_matrix() for _ in range(gate_set_size)])
    candidates_cache = jnp.array([
        jnp.pad(gate_set[candidates_ind_cache[G]], ((0, gate_set_size**G_max - gate_set_size**G), (0, G_max - G), (0, 0), (0, 0)))
        for G in G_list])
    arch_cache = jnp.array([
        jnp.pad(random_arch(n, G, focus_n), ((0, G_max - G), (0, 0)))
        for G in G_list])
    gates_cache = jnp.array([random_gates(candidates_cache[i][:gate_set_size**G_list[i]])[1] for i in range(len(G_list))])
    
    # junta learning
    relevant_qubits_cache = [
        get_relevant_qubits(n, None, arch_cache[i]) 
        for i in range(len(G_list))]
    n_relevant = jnp.array([len(relevant_qubits_cache[i]) for i in range(len(G_list))])
    n_relevant_max = int(n_relevant.max())
    arch_focus_cache = jnp.array([
        jnp.pad(focus_on_relevant_qubits(arch_cache[i], relevant_qubits_cache[i]), ((0, 0), (0, n_relevant_max - n_relevant[i])))
        for i in range(len(G_list))])
    
    # rotations_cache = jnp.array([random_unitary(2**n_relevant_max).to_matrix() for _ in range(sample_size_max)])
    rotations_cache = jnp.array([get_rotation(n_relevant_max, clifford=False, shallow=shallow) for _ in range(sample_size_max)])
    
    fidelity_cache = jnp.array([
        jnp.pad(
            fidelity_batch_candidates(n_relevant_max, G_list[i], arch_focus_cache[i][:G_list[i]], gates_cache[i][:G_list[i]], candidates_cache[i][:gate_set_size**G_list[i], :G_list[i]]),
            ((0, gate_set_size**G_max - gate_set_size**G_list[i]))
            )
        for i in range(len(G_list))])

    keys = jax.random.split(key, G_max+1)
    key = keys[0]
    keys = keys[1:]
    fid_list[run] = search_batch_G(jnp.array(G_list), keys, n_relevant_max, G_max, sample_size_max, gate_set_size, candidates_cache, arch_focus_cache, gates_cache, fidelity_cache, rotations_cache, shallow)

fid_mean = jnp.mean(fid_list, axis=0)
fid_med = jnp.quantile(fid_list, 0.5, axis=0)
fid_std = fid_list.std(axis=0)

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
plt.ylabel("Sample Size $N$")
plt.title(f"$n={n}$, {n_run} repetitions")
plt.legend()

f_name = f"n={n}_G={G_max}_nrun={n_run}"
if focus_n is not None:
    f_name += "_focus"
if shallow:
    f_name += "_shallow"

print('Saving into', f_name)
plt.savefig(f"{f_name}.pdf")
np.save(f"{f_name}.npy", fid_list)

plt.show()