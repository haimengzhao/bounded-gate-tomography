import numpy as np
import jax
import jax.numpy as jnp
from qiskit.quantum_info import random_clifford, random_unitary
import tensorcircuit as tc
K = tc.set_backend("jax")

from circuit import get_circ

I = jnp.array([[1, 0], [0, 1]])
X = jnp.array([[0, 1], [1, 0]])
Y = jnp.array([[0, -1j], [1j, 0]])
Z = jnp.array([[1, 0], [0, -1]])
bit_to_op = jnp.array([(I + Z) / 2, (I - Z) / 2])

depth_default = 10

def get_rotation(n, clifford=False, shallow=False, depth=depth_default):
    if not shallow:
        if clifford:
            return random_clifford(n).to_matrix()
        else:
            return random_unitary(2**n).to_matrix()
    else:
        if clifford:
            gate_per_layer = n // 2
            gates = np.array([random_clifford(2).to_matrix() for _ in range(gate_per_layer * depth)])
            return gates
        else:
            gate_per_layer = n // 2
            gates = np.array([random_unitary(4).to_matrix() for _ in range(gate_per_layer * depth)])
            return gates


def clifford_shadow(n, G, arch, gates, clifford, key, shallow):
    K.set_random_state(key)
    circ = get_circ(n, G, arch, gates)
    if not shallow:
        circ.any(*range(n), unitary=clifford)
    else:
        for i in range(depth_default):
            for j in range(n // 2):
                ind_1 = 2 * j + i % 2
                ind_2 = ind_1 + 1
                if ind_2 == n: # ind_2 corresponds to the n+1 th qubit
                    continue
                circ.any(ind_1, ind_2, unitary=clifford[i * (n // 2) + j])
    return circ.measure(*range(n))[0].astype(jnp.int32)

clifford_shadow_batch = jax.vmap(clifford_shadow, in_axes=(None, None, None, None, 0, 0, None))

def fidelity_estimate_single_shadow(obs_state_circ, basis, rotation, shallow):
    n = obs_state_circ._nqubits

    new_circ = obs_state_circ.copy()
    if not shallow:
        new_circ.any(*range(n), unitary=rotation)
    else:
        for i in range(depth_default):
            for j in range(n // 2):
                ind_1 = 2 * j + i % 2
                ind_2 = ind_1 + 1
                if ind_2 == n:
                    continue
                new_circ.any(ind_1, ind_2, unitary=rotation[i * (n // 2) + j])
    ops = [(bit_to_op[basis[i]], [i,]) for i in range(len(basis))]
    estimate = K.real((2**n + 1) * new_circ.expectation(*ops)) - 1

    return estimate

fidelity_estimate_batch_shadow = jax.vmap(fidelity_estimate_single_shadow, in_axes=(None, 0, 0, None))

def fidelity_estimate_single_candidate(n, G, arch, candidate, bases, rotations, shallow):
    circ = get_circ(n, G, arch, candidate)
    fidelity = fidelity_estimate_batch_shadow(circ, bases, rotations, shallow)
    return fidelity

fidelity_estimate_batch_candidate = jax.vmap(fidelity_estimate_single_candidate, in_axes=(None, None, None, 0, None, None, None))
