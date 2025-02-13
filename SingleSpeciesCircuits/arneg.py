# Prelim

import numpy as np

# ODEs

def Equ1(x, alpha, n):
    return ((alpha * x**(n-1)) / (1 + x**n)) - 1

# Sensitivity functions

def S_alpha_xss_analytic(xss, alpha, n):
    numer = 1 + xss**n
    denom = 1 - n + xss**n
    sensitivity = numer/denom
    return abs(sensitivity)

def S_n_xss_analytic(xss, alpha, n):
    numer = n * np.log(xss)
    denom = 1 - n + xss**n
    sensitivity = numer/denom
    return abs(sensitivity)

# Initial guesses

def generate_initial_guesses(beta_x_val, beta_y_val):
    return [
        np.array([0.5]),
        np.array([1]),
        np.array([1.5]),
        np.array([10])
    ]