# Prelim
import numpy as np

# ODEs
def Equ1(x, y, beta_x, n):
    return beta_x * y**n / (1 + y**n) - x
def Equ2(x, y, beta_y, n):
    return beta_y * x**n / (1 + x**n) - y

# Sensitivity functions
def S_betax_xss_analytic(xss, yss, n, beta_x, beta_y):
    hill = beta_y * xss**n / (1 + xss**n)
    numer = (1 + xss**n) * (1 + (hill)**n)
    denom = 1 - n**2 + (hill)**n + xss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)
def S_betax_yss_analytic(xss, yss, n, beta_x, beta_y):
    hill = beta_x * yss**n / (1 + yss**n)
    numer = n * (1 + yss**n)
    denom = 1 - n**2 + (hill)**n + yss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)
def S_betay_xss_analytic(xss, yss, n, beta_x, beta_y):
    hill = beta_y * xss**n / (1 + xss**n)
    numer = n * (1 + xss**n)
    denom = 1 - n**2 + (hill)**n + xss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)
def S_betay_yss_analytic(xss, yss, n, beta_x, beta_y):
    hill = beta_x * yss**n / (1 + yss**n)
    numer = (1 + yss**n) * (1 + (hill)**n)
    denom = 1 - n**2 + (hill)**n + yss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)
def S_n_xss_analytic(xss, yss, n, beta_x, beta_y):
    hill = beta_y * xss**n / (1 + xss**n)
    numer = n * (n * np.log(xss) + np.log(hill) + np.log(hill) * xss**n)
    denom = 1 - n**2 + (hill)**n + xss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)
def S_n_yss_analytic(xss, yss, n, beta_x, beta_y):
    hill = beta_x * yss**n / (1 + yss**n)
    numer = n * (n * np.log(yss) + np.log(hill) + np.log(hill) * yss**n)
    denom = 1 - n**2 + (hill)**n + yss**n * (1 + hill**n)
    sensitivity = numer/denom
    return abs(sensitivity)

# Initial guesses
def generate_initial_guesses(beta_x_val, beta_y_val):
    return [
        np.array([beta_x_val, beta_y_val]),
        np.array([beta_x_val / 2, beta_y_val / 2]),
        np.array([beta_x_val, 0]),
        np.array([beta_x_val / 2, 0]),
        np.array([0, beta_y_val]),
        np.array([0, beta_y_val / 2]),
        np.array([1, 1]),
        np.array([0.1, 0.1]),
    ]