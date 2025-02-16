# Prelim

import numpy as np

# ODEs

def Equ1(x, alpha, n):
    return (alpha / (1 + x**n)) - x

# Sensitivity functions

def S_alpha_xss_analytic(xss, alpha, n):
    numer = alpha * (1 + xss**n)
    denom = xss + alpha * n * xss**n + 2 * xss**(1+n) + xss**(1+2*n)
    sensitivity = numer/denom
    return abs(sensitivity)
    
def S_n_xss_analytic(xss, alpha, n):
    numer = alpha * n * np.log(xss) * xss**(n-1)
    denom = 1 + alpha * n * xss**(n-1) + 2 * xss**(n) + xss**(2*n)
    sensitivity = - numer/denom
    return abs(sensitivity)

# Initial guesses

def generate_initial_guesses(alpha_val, n_val):
    return [
        np.array([2]),
        np.array([0.5]),
        np.array([4.627])
    ]