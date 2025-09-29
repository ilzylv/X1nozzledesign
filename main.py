import numpy as np
import matplotlib.pyplot as plt
from src.funcoes_auxiliares import epsilon_k_razaoP2P1, empuxo
from src.atmosfera import us_standard_atmosphere
from src.tubeira_sino import gerar_tabela_pontos, plotar_completo, tubeira_sino

# Dados do motor X1
k = 1.22                    # Razão de calores específicos
R = 518                     # Constante específica do gás [J/kg*K]
F = 2.3 * 10**6             # Empuxo máximo (nível do mar) [N]
mdot = 1200                 # Vazão mássica [kg/s]
T1 = 3300                   # Temperatura na câmara de combustão [K]
P1 = 30 * 10**6             # Pressão na câmara de combustão [Pa]
Isp = 330                   # Impulso específico (nível do mar) [s]
At = 0.126                  # Área da garganta [m^2]]
Rt = np.sqrt(At/np.pi)      # Cálculo do raio da garganta
g = 9.80665                 # Aceleração da gravidade [m/s^2]

# Altitudes de operação
Ha = 0
Hb = 30e3
Hc = 80e3
Hd = 120e3

# Razões de expansão para cada estágio de voo
Ea = 30                     # Razão baixa para compensar alta pressão atmosférica
Eb = 60                    # Razão intermediária
Ec = 150                    # Razão alta justificada pela presença no vácuo

# Divisão em estágios
h1 = np.linspace(Ha, Hb)                # Fase 1
h2 = np.linspace(Hb, Hc)                # Fase 2
h3 = np.linspace(Hc, Hd, 100)      # Re-entrada

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

# Concatenação dos dados para plots combinados
h_total = np.concatenate((h1, h2, h3))
F_total = np.concatenate((F1, F2, F3))
P_atm_total = np.concatenate((P3_1, P3_2, P3_3))

# Plot das tubeiras
# Comprimento da tubeira (percentual)
l_camara_perc = 80

# Estágio 01
Rt_mm = Rt * 1000
angulos_a, contorno_a = tubeira_sino(k, Ea, Rt_mm, l_camara_perc)
plotar_completo(f'Tubeira estágio 1 (Razão de expansão = {Ea})', Rt, angulos_a, contorno_a, Ea)
gerar_tabela_pontos(contorno_a, 'tubeira_estagio_1.csv')

# Estágio 02
angulos_b, contorno_b = tubeira_sino(k, Eb, Rt_mm, l_camara_perc)
plotar_completo(f'Tubeira estágio 2 (Razão de expansão = {Eb})', Rt, angulos_b, contorno_b, Eb)
gerar_tabela_pontos(contorno_b, 'tubeira_estagio_2.csv')

# Estágio 03
angulos_c, contorno_c = tubeira_sino(k, Ec, Rt_mm, l_camara_perc)
plotar_completo(f'Tubeira estágio 3 (Razão de expansão = {Ec})', Rt, angulos_c, contorno_c, Ec)
gerar_tabela_pontos(contorno_c, 'tubeira_estagio_3.csv')

# Impulso específico para cada estágio
Isp1 = F1 / (mdot * g)
Isp2 = F2 / (mdot * g)
Isp3 = F3 / (mdot * g)
Isp_total = F_total / (mdot * g)

# Plot dos resultados
# Empuxo vs Altitude
plt.figure(figsize=(10, 6))
plt.plot(h / 1e3, F / 1e6, label='Empuxo')
plt.plot(h1 / 1e3, F1 / 1e6, label=f'Estágio 1 (E={Ea})')
plt.plot(h2 / 1e3, F2 / 1e6, label=f'Estágio 2 (E={Eb})')
plt.plot(h3 / 1e3, F3 / 1e6, label=f'Estágio 3 (E={Ec})')
plt.xlabel('Altitude (km)')
plt.ylabel('Empuxo (MN)')
plt.title('Empuxo do motor vs. Altitude')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Pressão de saída da tubeira vs Pressão ambiente
plt.figure(figsize=(12, 7))
plt.plot(h_total / 1e3, P_atm_total, label='Pressão atmosférica (ambiente)', color='k', linestyle='--')
plt.plot(h1 / 1e3, np.full_like(h1, P2a), label=f'Pressão de saída estágio 1 (P_exit = {P2a / 1e3:.1f} kPa)')
plt.plot(h2 / 1e3, np.full_like(h2, P2b), label=f'Pressão de saída estágio 2 (P_exit = {P2b / 1e3:.1f} kPa)')
plt.plot(h3 / 1e3, np.full_like(h3, P2c), label=f'Pressão de saída estágio 3 (P_exit = {P2c / 1e3:.1f} kPa)')
plt.yscale('log')
plt.xlabel('Altitude (km)', fontsize=12)
plt.ylabel('Pressão (Pa)', fontsize=12)
plt.title('Comparação entre pressão de saída da tubeira vs. Pressão ambiente', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Impulso específico vs Altitude
plt.figure(figsize=(10, 6))
plt.plot(h_total / 1e3, Isp_total, color='k', linewidth=2.5, label='Curva de desempenho')
plt.plot(h1 / 1e3, Isp1, label=f'Estágio 1 (E={Ea})')
plt.plot(h2 / 1e3, Isp2, label=f'Estágio 2 (E={Eb})')
plt.plot(h3 / 1e3, Isp3, label=f'Estágio 3 (E={Ec})')
plt.xlabel('Altitude (km)')
plt.ylabel('Impulso específico (s)')
plt.title('Impulso específico vs. Altitude')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Análise de pressão para o primeiro estágio
plt.figure(figsize=(12, 7))
plt.title(f'Análise de pressão para ε = {Ea}')
p_exit_1 = np.full_like(h1, P2a)
plt.plot(h1 / 1e3, P3_1, label='Pressão ambiente (Pa)', color='k', linestyle='--')
plt.plot(h1 / 1e3, p_exit_1, label=f'Pressão saída (ε={Ea})')
plt.fill_between(h1 / 1e3, p_exit_1, P3_1, where=p_exit_1 > P3_1, color='skyblue', alpha=0.6, label='Sub-expandido')
plt.fill_between(h1 / 1e3, p_exit_1, P3_1, where=p_exit_1 < P3_1, color='salmon', alpha=0.6, label='Super-expandido')
plt.ylabel('Pressão (Pa)')
plt.xlabel('Altitude (km)')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Análise de pressão para o segundo estágio
plt.figure(figsize=(12, 7))
plt.title(f'Análise de pressão para ε = {Eb}')
p_exit_2 = np.full_like(h2, P2b)
plt.plot(h2 / 1e3, P3_2, label='Pressão ambiente (Pa)', color='k', linestyle='--')
plt.plot(h2 / 1e3, p_exit_2, label=f'Pressão saída (ε={Eb})', color='darkorange')
plt.fill_between(h2 / 1e3, p_exit_2, P3_2, where=p_exit_2 > P3_2, color='skyblue', alpha=0.6)
plt.fill_between(h2 / 1e3, p_exit_2, P3_2, where=p_exit_2 < P3_2, color='salmon', alpha=0.6)
plt.ylabel('Pressão (Pa)')
plt.xlabel('Altitude (km)')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Análise de pressão para o terceiro estágio
plt.figure(figsize=(12, 7))
plt.title(f'Análise de pressão para ε = {Ec}')
p_exit_3 = np.full_like(h3, P2c)
plt.plot(h3 / 1e3, P3_3, label='Pressão ambiente (Pa)', color='k', linestyle='--')
plt.plot(h3 / 1e3, p_exit_3, label=f'Pressão saída (ε={Ec})', color='green')
plt.fill_between(h3 / 1e3, p_exit_3, P3_3, where=p_exit_3 > P3_3, color='skyblue', alpha=0.6)
plt.fill_between(h3 / 1e3, p_exit_3, P3_3, where=p_exit_3 < P3_3, color='salmon', alpha=0.6)
plt.ylabel('Pressão (Pa)')
plt.xlabel('Altitude (km)')
plt.yscale('log')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

# Encontra a curva de empuxo "ótima" (o máximo de cada ponto)
F_tubeira_A_em_h_total = empuxo(P1, At, k, P2a, P_atm_total, Ea)
F_tubeira_B_em_h_total = empuxo(P1, At, k, P2b, P_atm_total, Eb)
F_tubeira_C_em_h_total = empuxo(P1, At, k, P2c, P_atm_total, Ec)
plt.plot(h_total / 1e3, F_tubeira_A_em_h_total / 1e6, linestyle='--', label=f'Desempenho da tubeira 1 (ε={Ea})')
plt.plot(h_total / 1e3, F_tubeira_B_em_h_total / 1e6, linestyle='--', label=f'Desempenho da tubeira 2 (ε={Eb})')
plt.plot(h_total / 1e3, F_tubeira_C_em_h_total / 1e6, linestyle='--', label=f'Desempenho da tubeira 3 (ε={Ec})')
F_otimo = np.maximum.reduce([F_tubeira_A_em_h_total, F_tubeira_B_em_h_total, F_tubeira_C_em_h_total])
plt.plot(h_total / 1e3, F_otimo / 1e6, color='k', linewidth=2.5, label='Curva de empuxo ótima envelopada')
plt.title('Análise de desempenho para otimização das trocas', fontsize=16)
plt.xlabel('Altitude (km)')
plt.ylabel('Empuxo (MN)')
plt.grid(True, which='both')
plt.legend()
plt.show()