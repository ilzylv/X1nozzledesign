from scipy.optimize import fsolve, brentq
import numpy as np
from src.atmosfera import us_standard_atmosphere

def epsilon_k_razaoP2P1(epsilon, k):
    def mach_area_razao(Me):
        term1 = (k + 1) / 2
        term2 = 1 + (k - 1) / 2 * Me ** 2
        exponent = (k + 1) / (2 * (k - 1))

        area_ratio = (1 / Me) * ((term2 / term1) ** exponent)
        return epsilon - area_ratio

    try:
        # Usa bounds (do método) para mach supersônico
        Mach_saida = brentq(mach_area_razao, 1.001, 20.0)
    except ValueError:
        # Fallback para fsolve se brentq falhar
        Mach_saida = fsolve(mach_area_razao, x0=2.0)[0]

    # Calcula razão de pressão
    P_ratio = (1 + (k - 1) / 2 * Mach_saida ** 2) ** (-k / (k - 1))

    return P_ratio

def empuxo(P1, At, k, P_exit, P_ambient, E):
    # Termo de momento
    empuxo_momento = At * P1 * np.sqrt(
        ((2 * k**2) / (k - 1)) *
        (2 / (k + 1))**((k + 1) / (k - 1)) *
        (1 - (P_exit / P1)**((k - 1) / k))
    )
    # Termo de pressão
    empuxo_pressao = (P_exit - P_ambient) * E * At

    return empuxo_momento + empuxo_pressao

def epsilon_vs_altitude(h):
    if h < 30e3:
        return np.interp(h, [0, 30e3], [50, 200])
    elif h < 80e3:
        return np.interp(h, [30e3, 80e3], [200, 600])
    else:
        return np.interp(h, [80e3, 120e3], [600, 1000])