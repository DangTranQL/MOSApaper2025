# Prelim
import numpy as np

# ODEs
def Equ1(x, y, beta_x, n):
    return beta_x / (1 + y**n) - x
def Equ2(x, y, beta_y, n):
    return beta_y / (1 + x**n) - y

# Sensitivity functions
def S_betax_xss_analytic(xss, yss, n, beta_x, beta_y):
    frac = ( n**2 * beta_x * xss**(-1+n) * yss**(1+n) ) / ( beta_y * (1+yss**n)**2 )
    numer = beta_x
    denom = xss * (1 + yss**n) * (-1 + frac)
    sensitivity = - numer/denom
    return abs(sensitivity)
def S_betax_yss_analytic(xss, yss, n, beta_x, beta_y):
    numer = n * beta_x * beta_y * xss**(-1+n)
    denom = (1 + xss**n)**2 * yss - n**2 * beta_y * xss**n * yss**n + (1 + xss**n)**2 * yss**(1+n)
    sensitivity = - numer/denom 
    return abs(sensitivity)
def S_betay_xss_analytic(xss, yss, n, beta_x, beta_y):
    numer = n * beta_x * beta_y * yss**(-1+n)
    denom = - n**2 * beta_x * xss**n * yss**n + xss * (1 + yss**n)**2 + xss**(1+n) * (1 + yss**n)**2
    sensitivity = - numer/denom  
    return abs(sensitivity)
def S_betay_yss_analytic(xss, yss, n, beta_x, beta_y):
    frac = ( n**2 * beta_y * yss**(-1+n) * xss**(1+n) ) / ( beta_x * (1+xss**n)**2 )
    numer = beta_y
    denom = yss * (1 + xss**n) * (-1 + frac)
    sensitivity = - numer/denom 
    return abs(sensitivity)
def S_n_xss_analytic(xss, yss, n, beta_x, beta_y):
    numer = n * beta_x * (np.log(yss) + ( -n*np.log(xss) + np.log(yss) ) * xss**n) * yss**n
    denom = -n**2 * beta_x * xss**n * yss**n + xss * (1 + yss**n)**2 + xss**(1+n) * (1+yss**n)**2
    sensitivity = - numer/denom    
    return abs(sensitivity)
def S_n_yss_analytic(xss, yss, n, beta_x, beta_y):
    numer = n * beta_y * xss**n * (np.log(xss) + (np.log(xss) - n*np.log(yss)) * yss**n)
    denom = (1 + xss**n)**2 * yss - n**2 * beta_y * xss**n * yss**n + (1+xss**n)**2 * yss**(1+n)
    sensitivity = - numer/denom 
    return abs(sensitivity)

# Initial guesses
def generate_initial_guesses(beta_x_val, beta_y_val):
    return [
        np.array([beta_x_val, 0]),
        np.array([0, beta_y_val]),
        np.array([beta_x_val / 2, beta_y_val / 2]),
        np.array([beta_x_val, beta_y_val]),
        np.array([0, 0]),
        np.array([1, 1])
    ]