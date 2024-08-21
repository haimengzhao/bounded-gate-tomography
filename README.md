# Learning quantum states and unitaries of bounded gate complexity
This repo contains the code for the paper [Learning quantum states and unitaries of bounded gate complexity](https://arxiv.org/abs/2310.19882).

The code implements the state learning algorithm detailed in the paper and study how its sample complexity $N$ scales with the gate size $G$.

Thanks to the features of [TensorCircuit](https://github.com/tencent-quantum-lab/tensorcircuit) and [JAX](https://github.com/google/jax), our implementation support automatic vectorization and just-in-time (JIT) compilation.
It can be executed on CPU, GPU, or TPU.

## Requirements
The code is written in Python3 and built upon [TensorCircuit](https://github.com/tencent-quantum-lab/tensorcircuit) and [JAX](https://github.com/google/jax).
To execute it, we need to first install `numpy`, `jax`, `tensorcircuit`, `qiskit`, `tqdm`. To plot figures, we also need `matplotlib` and `SciencePlots`.
They can all be installed using PyPI.

## File Structure
`circuit.py` contains elementary circuit functions including generating random circuit architectures and gate sequences, simulate circuits, computing fidelity, and junta learning.

`shadow.py` implements Clifford classical shadow using JAX and TensorCircuit.
One can choose to replace Clifford gates with Haar random unitaries to reduce statistical fluctuation.
We also implement "shallow shadows" where global Clifford rotations are replaced by brickwork Clifford circuits with depth $\Theta(\log n)$.
See [shallow shadows paper](https://arxiv.org/abs/2209.12924) and [shallow pseudorandom unitary paper](https://arxiv.org/abs/2407.07754) for details.

`learn.py` implements the learning algorithm.

`test.py` contains the pipeline for mass production that reproduces the data in the paper.

`plot.py` plots the figures.