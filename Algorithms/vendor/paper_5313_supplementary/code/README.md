# Fast Asymptotically Optimal Algorithms for Non-Parametric Stochastic Bandits
## Code base to reproduce experiments

This folder contains python files to reproduce figures and tables of the associated paper.

## Installing requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install packages listed in `requirements.txt`.
Please create and activate a virtual environment if necessary.

```bash
pip install -r requirements.txt
```

## Usage

To plot a given Figure or Table, run the corresponding python file, e.g.,
```bash
python plot_figure_3_bernoulli_xp.py
```
**Warning**: By default, the code will be executed on **as many CPUs as there are available**.
If you want to specify the number of used CPUs, please modify the `num_cores` argument of the `experiment`.
For instance,
```python
experiment = Experiment(algorithms, bandit, suffix=" figure 3", num_cores=2)
```
instead of
```python
experiment = Experiment(algorithms, bandit, suffix=" figure 3")
```
as written l. 32 of `./plot_figure_3_bernoulli_xp.py`.

Results of the experiment can then be found in the folder `./results/corresponding_experiment`.
Each python file start with a specification of the `horizon` and number of experiments `nbr_xp` used.
To reduce the computation time, we set `nbr_xp = 50`.
```python
horizon = 10000
nbr_xp = 50
```
You can easily increase the number of runs to reduce the noise by modifying this parameter.

## Structure
- `arms.py` contains the implementation of the mathematical object **arm**, mainly, a random variable
- `bandits.py` contains the implementation of the mathematical object **arm**, mainly, a collection of random variables
- `learning_algorithms.py` contains the implementation of all tested sequential learning algorithms
- `experiments.py` contains the code that handle running experiments in parallel, storing and plotting the results
- `utils.py` contains generic function that are used in other files such as computing an `argmin`, the `kl-divergence`...
