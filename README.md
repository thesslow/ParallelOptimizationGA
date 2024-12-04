

# Parallel Genetic Algorithm for Fitness Evaluation

This repository implements a parallelized version of a Genetic Algorithm (GA) for optimizing hyperparameters in machine learning models. The core focus is on using parallel processing to speed up fitness evaluations during the evolutionary process, specifically targeting fitness score evaluations using cross-validation.

## Features

- **Parallelized Population Creation**: The initial population of individuals is generated in parallel across multiple processes to speed up the process.
- **Parallelized Fitness Evaluation**: The fitness of each individual in the population is evaluated in parallel, leveraging multiple CPU cores.
- **Genetic Algorithm Process**: The algorithm uses a typical genetic approach with **selection**, **crossover**, and **mutation** to evolve the population over several generations.
- **Scikit-learn Integration**: Uses Scikit-learn's `cross_val_score` for fitness evaluation of models, ensuring that the solution is based on sound machine learning practices.

## Installation

To use this code, clone this repository and install the required dependencies.

```bash
git clone https://github.com/your-username/ParallelOptimizationGA.git
cd ParallelOptimizationGA
pip install -r requirements.txt
