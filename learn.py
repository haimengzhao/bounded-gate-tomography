import jax
import jax.numpy as jnp
import tensorcircuit as tc
K = tc.set_backend("jax")

from shadow import fidelity_estimate_batch_candidate, clifford_shadow_batch

def exhaustive_search(n, G, bases, rotations, arch, candidates, shallow):
    
    fids = fidelity_estimate_batch_candidate(n, G, arch, candidates, bases, rotations, shallow).mean(axis=-1)
    
    best_index = fids.argmax()
    best_fidelity = fids[best_index]
    best_circ = candidates[best_index]
    
    return best_fidelity, best_index, best_circ

def search(n, fid_est, fid_cache):
    fids = fid_est.mean(axis=-1)
    best_index = fids.argmax()
    best_fidelity = fids[best_index]
    return fid_cache[best_index]
search = jax.jit(search, static_argnums=(0))

def search_single_samplesize(sample_size, sample_size_max, fid_est, fid_cache):
    fid_est_mask = jnp.where(jnp.arange(sample_size_max) < sample_size, fid_est, 0)
    fid_est_mask = jnp.sum(fid_est_mask, axis=1) / sample_size
    best_index = fid_est_mask.argmax()
    return fid_cache[best_index]

search_batch_samplesize = jax.vmap(search_single_samplesize, in_axes=(0, None, None, None))

def search_single_G(G, key, n, G_max, sample_size_max, gate_set_size, candidates_cache, arch_cache, gates_cache, fidelity_cache, rotations_cache, shallow):

    candidates = candidates_cache[G-1]
    arch = arch_cache[G-1]
    gates = gates_cache[G-1]
    fid_cache = fidelity_cache[G-1]

    keys = jax.random.split(key, sample_size_max)
    bases = clifford_shadow_batch(n, G, arch, gates, rotations_cache, keys, shallow)
    
    fid_est = fidelity_estimate_batch_candidate(n, G, arch, candidates, bases, rotations_cache, shallow)
    mask = jnp.arange(gate_set_size**G_max) < gate_set_size**G
    fid_est = jnp.where(mask.reshape(-1, 1), fid_est, -1)

    sample_size_list = jnp.arange(1, sample_size_max+1)
    fid_sublist = search_batch_samplesize(sample_size_list, sample_size_max, fid_est, fid_cache)

    return fid_sublist

search_batch_G = jax.vmap(search_single_G, in_axes=(0, 0, None, None, None, None, None, None, None, None, None, None))

search_batch_G = jax.jit(search_batch_G, static_argnums=(2, 3, 4, 5, 11))
