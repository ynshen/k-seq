"""Function for variant pool design"""
from math import factorial


def combination(n, k):
    return factorial(n) / (factorial(n - k) * factorial(k))


def num_of_seq(d, length=21, letter_book_size=4):
    return int(combination(length, d) * (letter_book_size - 1) ** d)


def d_mutant_fraction(d, mutation_rate, length=21, letter_book_size=4):
    """Relative abundance for a single d-order mutants"""

    return (1 - mutation_rate) ** (length - d) * (mutation_rate / (letter_book_size - 1)) ** d


def neighbor_effect_observation(xi, d, eta=0.09, L=21):
    """Get the ratio of observed abundance for a d-th order mutant,
    considering the neighbor effect under given sequencing error rate (xi)"""

    phi = 3 * (1 - eta) / eta
    rho = (d * phi + 2 * d + (3 * L - 3 * d) / phi) * (1 - xi) ** (L - 1) * xi / 3 + (1 - xi) ** L
    return rho


def neighbor_effect_error(xi, d, eta=0.09, L=21):
    rho = neighbor_effect_observation(xi=xi, d=d, eta=eta, L=L)
    return 1 - ((1 - xi) ** L) / rho