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


import numpy as np
from src.funcoes_auxiliares import epsilon_k_razaoP2P1
from src.atmosfera import us_standard_atmosphere


def encontrar_estagios(k, P1, At, h_max, n_pontos=300, tolerancia=0.05):

    # Determina as altitudes de troca de estágio e as razões de expansão ótimas (P2 = 0.6 * P3)
    altitudes = np.linspace(0, h_max, n_pontos)
    P3 = us_standard_atmosphere(altitudes)["P"]

    eps_otimos = []
    for i, Pamb in enumerate(P3):
        melhor_eps = None
        erro_min = 1e9

        for eps in np.linspace(5, 1000, 500):
            P2 = P1 * epsilon_k_razaoP2P1(eps, k)
            alvo = 0.6 * Pamb
            erro = abs(P2 - alvo) / alvo

            if erro < erro_min:
                erro_min = erro
                melhor_eps = eps

        if erro_min <= tolerancia:
            eps_otimos.append((altitudes[i], melhor_eps))

    # Determinar pontos de troca
    estagios = []
    if eps_otimos:
        atual_eps = eps_otimos[0][1]
        estagios.append((0, atual_eps))  # início no nível do mar

        for h, eps in eps_otimos:
            if abs(eps - atual_eps) / atual_eps > 0.3:  # diferença grande (~30%)
                estagios.append((h, eps))
                atual_eps = eps

    return estagios