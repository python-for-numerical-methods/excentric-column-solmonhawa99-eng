import numpy as np
from scipy.optimize import bisect


def find_critical_load(L, E, A, r, c, e, sigma_allow):

    def equation(P):
        sec_term = 1 / np.cos(
            (L / (2 * r)) * np.sqrt(P / (E * A))
        )

        sigma_max = (P / A) * (
            1 + (e * c / r**2) * sec_term
        )

        return sigma_max - sigma_allow

    P_low = 0.0
    P_high = sigma_allow * A

    while equation(P_high) < 0:
        P_high *= 2

    return float(bisect(equation, P_low, P_high))
