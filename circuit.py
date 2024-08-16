import numpy as np
import jax
import jax.numpy as jnp
import tensorcircuit as tc
K = tc.set_backend("jax")

I = jnp.array([[1, 0], [0, 1]])
X = jnp.array([[0, 1], [1, 0]])
Y = jnp.array([[0, -1j], [1j, 0]])
Z = jnp.array([[1, 0], [0, -1]])
bit_to_op = jnp.array([(I + Z) / 2, (I - Z) / 2])

def random_arch(n, G, focus_n=None):
    if focus_n is not None:
        ind = np.random.choice(focus_n-1, size=(G,))
    else:
        ind = np.random.choice(n-1, size=(G,))
    arch = np.zeros((G, n-1), dtype=np.int32)
    arch[np.arange(G), ind] = 1
    arch = jnp.array(arch)

    return arch

def enumerate_gates(gate_set_size, G):
    # all possible gate sequences of length G from gate_set
    # use meshgrid
    ind = jnp.array(jnp.meshgrid(*[jnp.arange(gate_set_size) for _ in range(G)])).T.reshape(-1, G)
    return ind

def random_gates(candidates):
    length = len(candidates)
    ind = np.random.randint(length)
    return ind, candidates[ind]

def get_circ(n, G, arch, gates):
    circ = tc.Circuit(n)

    gates_id = jnp.array([jnp.eye(4) for _ in range(len(gates))])
    gates_full = jnp.stack([gates_id, gates], axis=0)
    
    for i in range(len(gates)):
        for j in range(n-1):
            circ.any(j, j+1, unitary=gates_full[arch[i][j]][i])

    return circ

def fidelity(n, G, arch, gates_true, gates_pred):
    state_true = get_circ(n, G, arch, gates_true)

    gates_id = jnp.array([jnp.eye(4) for _ in range(G)])
    gates_full = jnp.stack([gates_id, gates_pred[:G][::-1]], axis=0)
    
    for i in range(G):
        for j in range(n-1):
            state_true.any(j, j+1, unitary=gates_full[arch[:G][::-1][i][j]][i].conj().T)
    
    ops = [(bit_to_op[0], [i,]) for i in range(n)]
    fid = jnp.real(state_true.expectation(*ops))
    return fid

fidelity_batch_candidates = jax.jit(jax.vmap(fidelity, in_axes=(None, None, None, None, 0)), static_argnums=(0, 1))

def get_relevant_qubits(n, circ, arch=None, n_shots=10):
    if arch is not None:
        arch_pos = arch.argmax(axis=-1)
        arch_pos = np.stack([arch_pos, arch_pos+1])
        return np.unique(arch_pos)
    else:
        qubits = []
        for _ in range(n_shots):
            res = circ.measure(*list(range(4)))[0].astype(int)
            res = np.where(res == 1)[0]
            qubits.extend(res)
        qubits = np.array(qubits)
        return np.unique(qubits)

def focus_on_relevant_qubits(arch, relevant_qubits):
    G = arch.shape[0]
    arch_pos = arch.argmax(axis=-1)
    ind_all_zero_rows = np.sum(arch, axis=-1) == 0
    reverse_map = {int(i): j for j, i in enumerate(relevant_qubits)}
    arch_pos = np.vectorize(reverse_map.get)(arch_pos)
    arch_new = np.zeros((G, len(relevant_qubits)-1), dtype=np.int32)
    arch_new[np.arange(G), arch_pos] = 1
    arch_new[ind_all_zero_rows] = 0
    return arch_new