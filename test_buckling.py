import numpy as np
from scipy.optimize import bisect

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    """
    Finds the critical load P using the Secant Formula.
    L, E, A, r, c, e, sigma_allow are the input parameters.
    """
    
    # Define the target function f(P) = sigma_max - sigma_allow
    # We are looking for the root where sigma_max matches allowable stress.
    def f(P):
        # Secant Formula: sigma_max = (P/A) * [1 + (ec/r^2) * sec( (L/2r) * sqrt(P/EA) )]
        # sec(x) = 1/cos(x)
        
        # Calculate the argument for the cosine function
        arg = (L / (2 * r)) * np.sqrt(P / (E * A))
        
        # Calculate sigma_max based on the formula provided in Screenshot 2026-05-06 171250.png
        sigma_max = (P / A) * (1 + (e * c / r**2) * (1 / np.cos(arg)))
        
        return sigma_max - sigma_allow

    # Numerical Search Bounds:
    # The argument (L/2r)*sqrt(P/EA) must be less than pi/2 to avoid division by zero.
    # Solving for P gives the theoretical Euler limit:
    p_euler = (np.pi**2 * E * A * r**2) / (L**2)
    
    # The load P also cannot exceed the basic yield load: sigma_allow * A
    p_yield = sigma_allow * A
    
    # We set the search range between a near-zero value and the lower of the two limits.
    # We use 0.99 to stay safely away from the vertical asymptote.
    low = 1e-6
    high = min(p_euler, p_yield) * 0.999
    
    try:
        # Perform bisection search as suggested in Screenshot 2026-05-06 171314.png
        return bisect(f, low, high, xtol=1e-5)
    except ValueError:
        # Fallback in case the limits don't bracket the root perfectly
        return high
