import numpy as np
def us_standard_atmosphere(h):
    """
    Modelo simplificado da Atmosfera Padrão dos EUA (0–120 km).
    Retorna T [K], P [Pa], rho [kg/m³] para uma altitude h [m].
    """

    # garantir array numpy
    h = np.atleast_1d(h).astype(float)

    # Definição das camadas: (h_base [m], T_base [K], gradiente L [K/m])
    camadas = [
        (0, 288.15, -0.0065),  # Troposfera
        (11000, 216.65, 0.0),  # Tropopausa
        (20000, 216.65, 0.001),  # Estratosfera baixa
        (32000, 228.65, 0.0028),  # Estratosfera alta
        (47000, 270.65, 0.0),  # Estratopausa
        (51000, 270.65, -0.0028),  # Mesosfera baixa
        (71000, 214.65, -0.002),  # Mesosfera alta
        (84852, 186.95, 0.0)  # até ~86 km (acima disso extrapola exponencialmente)
    ]

    # Constantes
    R = 287.05
    g0 = 9.80665
    P0 = 101325.0  # pressão ao nível do mar

    # Arrays de saída
    T = np.zeros_like(h)
    P = np.zeros_like(h)

    # valores de referência
    T_base = camadas[0][1]
    h_base = camadas[0][0]
    L = camadas[0][2]

    for i, (h_b, T_b, L) in enumerate(camadas):
        h_top = camadas[i + 1][0] if i + 1 < len(camadas) else 120000.0
        mask = (h >= h_b) & (h < h_top)

        if L == 0.0:  # camada isotérmica
            T[mask] = T_b
            P[mask] = P0 * np.exp(-g0 * (h[mask] - h_b) / (R * T_b))
        else:  # camada com gradiente
            T[mask] = T_b + L * (h[mask] - h_b)
            P[mask] = P0 * (T[mask] / T_b) ** (-g0 / (R * L))

        # Atualiza condições de base para próxima camada
        if i < len(camadas) - 1:
            h_next, T_next, L_next = camadas[i + 1]
            if L == 0.0:
                P0 = P0 * np.exp(-g0 * (h_next - h_b) / (R * T_b))
            else:
                P0 = P0 * (T_next / T_b) ** (-g0 / (R * L))

    rho = P / (R * T)
    return {"T": T, "P": P, "rho": rho}
