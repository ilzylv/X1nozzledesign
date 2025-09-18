import numpy as np

def us_standard_atmosphere(h):
    g0 = 9.80665        # [m/s^2]
    R = 8.31432         # [J/(mol*K)]
    M = 0.0289644       # [kg/mol]
    P0 = 101325.0       # Pressão a nível do mar [Pa]
    T0 = 288.15         # Temperatura nível do mar [K]

    # Camadas até 120 km (base, T_b, gradiente L [K/m])
    camadas = [
        (0,     288.15, -0.0065),   # 0–11 km
        (11000, 216.65,  0.0),      # 11–20 km
        (20000, 216.65,  0.001),    # 20–32 km
        (32000, 228.65,  0.0028),   # 32–47 km
        (47000, 270.65,  0.0),      # 47–51 km
        (51000, 270.65, -0.0028),   # 51–71 km
        (71000, 214.65, -0.002),    # 71–86 km
        (86000, 186.95,  0.0),      # 86–91 km
        (91000, 186.95,  0.004),    # 91–120 km
    ]

    h = np.atleast_1d(h).astype(float)
    P = np.zeros_like(h)
    T = np.zeros_like(h)

    # Condições iniciais
    Pb, Tb, hb = P0, T0, 0.0

    for i, (h_base, T_base, L) in enumerate(camadas):
        # Altitude máxima da camada
        h_top = camadas[i+1][0] if i+1 < len(camadas) else 120000

        # Máscara de pontos dentro da camada (garante que está na camada condizente)
        mask = (h >= h_base) & (h < h_top)
        if not np.any(mask):
            # Atualiza pressão base para próxima camada
            if L == 0:
                Pb = Pb * np.exp(-g0 * M * (h_top-hb) / (R*Tb))
            else:
                Tb_new = Tb + L * (h_top-hb)
                Pb = Pb * (Tb_new/Tb) ** (-g0*M/(R*L))
                Tb = Tb_new
            hb = h_top
            continue

        if L == 0:  # isotérmica
            T[mask] = Tb
            P[mask] = Pb * np.exp(-g0 * M * (h[mask]-hb) / (R*Tb))
        else:       # gradiente
            T[mask] = Tb + L*(h[mask]-hb)
            P[mask] = Pb * (T[mask]/Tb) ** (-g0*M/(R*L))

    # Densidadepip
    rho = (P * M) / (R * T)

    return {"P": P, "T": T, "rho": rho}
