import numpy as np
import matplotlib.pyplot as plt
from src.funcoes_auxiliares import epsilon_k_razaoP2P1, empuxo, encontrar_estagios
from src.atmosfera import us_standard_atmosphere

# Dados do motor X1
k = 1.22 # Razão de calores específicos
R = 518 # Constante específica do gás [J/kg*K]
F = 2.3 * 10**6 # Empuxo máximo (nível do mar) [N]
mdot = 1200 # Vazão mássica [kg/s]
T1 = 3300 # Temperatura na câmara de combustão [K]
P1 = 30 * 10**6 # Pressão na câmara de combustão [Pa]
Isp = 330 # Impulso específico (nível do mar) [s]
At = 0.126 # Área da garganta [m^2]

# Altitudes de operação
estagios = encontrar_estagios(k, P1, At, 120 * 10 **3)
print("\n estágios otimos")
for i, (h, eps) in enumerate(estagios, start=1):
    print(f"Estágio {i}: troca em {h/1000:.1f} km → epislon ótimo = {eps:.2f}")

Ha = 0
Hb = 30e3
Hc = 80e3
Hd = 120e3

# Razões de expansão para cada estágio de voo
Ea = 22
Eb = 88
Ec = 102

# Divisão em estágios
h1 = np.linspace(Ha, Hb)      # Fase 1
h2 = np.linspace(Hb, Hc)      # Fase 2
h3 = np.linspace(Hc, Hd, 100) # Re-entrada

# Pressões de saída de acordo com a razão de pressões
P2a = P1 * epsilon_k_razaoP2P1(Ea, k)
P2b = P1 * epsilon_k_razaoP2P1(Eb, k)
P2c = P1 * epsilon_k_razaoP2P1(Ec, k)

# Pressões atmosféricas para as altitudes de cada estágio de voo
atm1 = us_standard_atmosphere(h1)
atm2 = us_standard_atmosphere(h2)
atm3 = us_standard_atmosphere(h3)

P3_1 = atm1["P"]
P3_2 = atm2["P"]
P3_3 = atm3["P"]

# Cálculo do empuxo para cada estágio de voo
F1 = empuxo(P1, At, k, P2a, P3_1, Ea)
F2 = empuxo(P1, At, k, P2b, P3_2, Eb)
F3 = empuxo(P1, At, k, P2c, P3_3, Ec)

# Concatenando as arrays para plot
h = np.concatenate((h1, h2, h3))
F = np.concatenate((F1, F2, F3))

# Plot dos resultados
plt.figure(figsize=(10, 6))
plt.plot(h / 1e3, F / 1e6, label='Empuxo')
plt.xlabel('Altitude (km)')
plt.ylabel('Empuxo (MN)')
plt.title('Empuxo do motor vs. Altitude para diferentes razões de expansão')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

