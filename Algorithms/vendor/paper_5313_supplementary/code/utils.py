import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


########################################
#              Plot Bandit             #
########################################
def plot_bandit(bandit):
    plt.figure(figsize=(12, 7))
    plt.xticks(range(0, bandit.nbr_arms), range(0, bandit.nbr_arms))
    plt.plot(range(0, bandit.nbr_arms), bandit.rewards)
    plt.scatter(range(0, bandit.nbr_arms), bandit.rewards)
    plt.show()


########################################
#           KL Divergences             #
########################################
def kinf(r_list, r_max, upper_bound=1.):
    if np.isclose(r_max, upper_bound):
        upper_bound = upper_bound + 1e-8
    x = np.array(r_list)

    # print("kinf info", r_max, upper_bound, np.min(x), x)

    def dual(u):
        return - np.mean(np.log(1 - (x - r_max) * u))

    ret = minimize_scalar(
        dual,
        method="bounded",
        bounds=(0, 1.0 / (upper_bound - r_max)),
    )
    return ret


def online_kinf(r_list, r_max_list, upper_bound=1.):
    x = np.array(r_list)
    y = np.array(r_max_list)

    def dual(u):
        return - np.mean(np.log(1 - ((x - y) / (upper_bound - y)) * u))

    ret = minimize_scalar(
        dual,
        method="bounded",
        bounds=(0, 1.0),
    )
    return ret


def kl_ucb_bern(mean, n, t, eps=1e-9):
    a = mean
    b = 1.
    threshold = (np.log(t) + 3 * np.log(max(1., np.log(t)))) / n
    c = (a + b) / 2
    kl = kl_bernoulli(mean, c)
    stop = (np.abs(threshold - kl) <= eps) or ((b - a) <= eps)
    while not stop:
        if kl <= threshold:
            a = c
        else:
            b = c
        c = (a + b) / 2
        kl = kl_bernoulli(mean, c)
        stop = ((threshold - kl) <= eps) or ((b - a) <= eps)
    return c


def kl_ucb(samples, mean, n, t, eps=1e-9):
    a = mean
    b = 1.
    threshold = (np.log(t) + 3 * np.log(max(1., np.log(t)))) / n
    c = (a + b) / 2
    kl = - kinf(samples, c).fun
    stop = (np.abs(threshold - kl) <= eps) or ((b - a) <= eps)
    while not stop:
        if kl <= threshold:
            a = c
        else:
            b = c
        c = (a + b) / 2
        kl = - kinf(samples, c).fun
        stop = ((threshold - kl) <= eps) or ((b - a) <= eps)
    return c


def kl_bernoulli(mean_1, mean_2, eps=1e-15):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = np.minimum(np.maximum(mean_1, eps), 1 - eps)
    y = np.minimum(np.maximum(mean_2, eps), 1 - eps)
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))


def multinomial_kinf(count, support, r_max, upper_bound=1.):
    if np.isclose(r_max, upper_bound):
        upper_bound = upper_bound + 1e-8

    def dual(u):
        return - np.sum(count * np.log(1 - (support - r_max) * u))

    ret = minimize_scalar(
        dual,
        method="bounded",
        bounds=(0, 1.0 / (upper_bound - r_max)),
    )
    return ret


def kl_gaussian(mean_1, mean_2, sig2=1.):
    """Kullback-Leibler divergence for Gaussian distributions."""
    return ((mean_1 - mean_2) ** 2) / (2 * sig2)


########################################
#         Argument selectors           #
########################################
def randamax(v, t=None, i=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax
    """
    if i is None:
        idxs = np.where(v == np.amax(v))[0]
        if t is None:
            idx = np.random.choice(idxs)
        else:
            assert len(v) == len(t), f"Lengths should match: len(V)={len(v)} - len(T)={len(t)}"
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(v[i] == np.amax(v[i]))[0]
        if t is None:
            idx = i[np.random.choice(idxs)]
        else:
            assert len(v) == len(t), f"Lengths should match: len(V)={len(v)} - len(T)={len(t)}"
            t = t[i]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = i[idxs[t_idxs]]
    return idx


def randamin(v, t=None, i=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax
    """
    if i is None:
        idxs = np.where(v == np.amin(v))[0]
        if t is None:
            idx = np.random.choice(idxs)
        else:
            assert len(v) == len(t), f"Lengths should match: len(V)={len(v)} - len(T)={len(t)}"
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(v[i] == np.amin(v[i]))[0]
        if t is None:
            idx = i[np.random.choice(idxs)]
        else:
            assert len(v) == len(t), f"Lengths should match: len(V)={len(v)} - len(T)={len(t)}"
            t = t[i]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = i[idxs[t_idxs]]
    return idx
