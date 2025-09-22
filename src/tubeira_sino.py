import matplotlib.pyplot as plt
from bisect import bisect_left
import numpy as np
import pandas as pd

"""
Baseado nas notas técnicas: "The thrust optimised parabolic nozzle"
http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf
"""

def tubeira_sino(k, arazao, Rt, l_camara):
    """
    Calcula o contorno de uma tubeira em formato de sino.
    Parâmetros:
    k: razão de calores específicos
    arazao: razão de área (Ae/At)
    Rt: raio da garganta
    l_camara: percentual do comprimento da tubeira (60, 80, 90)
    """
    entrant_angulo = -135  # Ângulo de entrada, tipicamente -135°
    ea_radiano = np.radians(entrant_angulo)

    # Percentual de comprimento da tubeira
    if l_camara == 60:
        Lnp = 0.6
    elif l_camara == 80:
        Lnp = 0.8
    elif l_camara == 90:
        Lnp = 0.9
    else:
        Lnp = 0.8

    # Encontrar os ângulos das paredes
    angulos = angulos_paredes(arazao, Rt, l_camara)
    comprimento_tubeira = angulos[0]
    theta_n = angulos[1]
    theta_e = angulos[2]

    intervalo = 100

    # Seção de entrada da garganta (throat entrant section)
    ea_comeco = ea_radiano
    ea_fim = -np.pi/2
    lista_angulos = np.linspace(ea_comeco, ea_fim, intervalo)
    xe = []
    ye = []
    for i in lista_angulos:
        xe.append(1.5 * Rt * np.cos(i))
        ye.append(1.5 * Rt * np.sin(i) + 2.5 * Rt)

    # Seção de saída da garganta (throat exit section)
    ea_comeco = -np.pi/2
    ea_fim = theta_n - np.pi/2
    lista_angulos = np.linspace(ea_comeco, ea_fim, intervalo)
    xe2 = []
    ye2 = []
    for i in lista_angulos:
        xe2.append(0.382 * Rt * np.cos(i))
        ye2.append(0.382 * Rt * np.sin(i) + 1.382 * Rt)

    # Seção do sino (bell section) - Curva quadrática de Bézier
    # Ponto N
    Nx = 0.382 * Rt * np.cos(theta_n - np.pi/2)
    Ny = 0.382 * Rt * np.sin(theta_n - np.pi/2) + 1.382 * Rt

    # Ponto E (saída)
    Ex = Lnp * ((np.sqrt(arazao) - 1) * Rt) / np.tan(np.radians(15))
    Ey = np.sqrt(arazao) * Rt

    # Gradientes
    m1 = np.tan(theta_n)
    m2 = np.tan(theta_e)

    # Interseções
    C1 = Ny - m1 * Nx
    C2 = Ey - m2 * Ex

    # Ponto Q (interseção das linhas)
    Qx = (C2 - C1) / (m1 - m2)
    Qy = (m1 * C2 - m2 * C1) / (m1 - m2)

    # Curva quadrática de Bézier
    lista_parametros = np.linspace(0, 1, intervalo)
    xsino = []
    ysino = []
    for t in lista_parametros:
        xsino.append(((1-t)**2) * Nx + 2*(1-t)*t * Qx + (t**2) * Ex)
        ysino.append(((1-t)**2) * Ny + 2*(1-t)*t * Qy + (t**2) * Ey)

    # Criar valores negativos para a outra metade da tubeira
    nye = [-y for y in ye]
    nye2 = [-y for y in ye2]
    nysino = [-y for y in ysino]

    return angulos, (xe, ye, nye, xe2, ye2, nye2, xsino, ysino, nysino)


def angulos_paredes(ar, Rt, l_camara=80):
    """
    Encontra os ângulos das paredes (theta_n, theta_e) para uma dada razão de área

    Parâmetros:
    ar: razão de área
    Rt: raio da garganta
    l_camara: percentual do comprimento (60, 80, 90)
    """
    # Dados empíricos dos ângulos das paredes
    arazao = [4, 5, 10, 20, 30, 40, 50, 100]
    theta_n_60 = [26.5, 28.0, 32.0, 35.0, 36.2, 37.1, 35.0, 40.0]
    theta_n_80 = [21.5, 23.0, 26.3, 28.8, 30.0, 31.0, 31.5, 33.5]
    theta_n_90 = [20.0, 21.0, 24.0, 27.0, 28.5, 29.5, 30.2, 32.0]
    theta_e_60 = [20.5, 20.5, 16.0, 14.5, 14.0, 13.5, 13.0, 11.2]
    theta_e_80 = [14.0, 13.0, 11.0, 9.0, 8.5, 8.0, 7.5, 7.0]
    theta_e_90 = [11.5, 10.5, 8.0, 7.0, 6.5, 6.0, 6.0, 6.0]

    # Comprimento da tubeira
    f1 = ((np.sqrt(ar) - 1) * Rt) / np.tan(np.radians(15))

    if l_camara == 60:
        theta_n = theta_n_60
        theta_e = theta_e_60
        Ln = 0.6 * f1
    elif l_camara == 80:
        theta_n = theta_n_80
        theta_e = theta_e_80
        Ln = 0.8 * f1
    elif l_camara == 90:
        theta_n = theta_n_90
        theta_e = theta_e_90
        Ln = 0.9 * f1
    else:
        theta_n = theta_n_80
        theta_e = theta_e_80
        Ln = 0.8 * f1

    # Encontrar o índice mais próximo na lista de razões de área
    x_index, x_val = encontrar_mais_proximo(arazao, ar)

    # Se o valor no índice for próximo ao input, retorna diretamente
    if round(arazao[x_index], 1) == round(ar, 1):
        return Ln, np.radians(theta_n[x_index]), np.radians(theta_e[x_index])

    # Interpolação linear para valores intermediários
    if x_index > 2:
        ar_meio = arazao[x_index-2:x_index+2]
        theta_n_meio = theta_n[x_index-2:x_index+2]
        theta_e_meio = theta_e[x_index-2:x_index+2]
        tn_valor = interpolar(ar_meio, theta_n_meio, ar)
        te_valor = interpolar(ar_meio, theta_e_meio, ar)
    elif (len(arazao) - x_index) <= 1:
        ar_meio = arazao[x_index-2:len(arazao)]
        theta_n_meio = theta_n[x_index-2:len(theta_n)]
        theta_e_meio = theta_e[x_index-2:len(theta_e)]
        tn_valor = interpolar(ar_meio, theta_n_meio, ar)
        te_valor = interpolar(ar_meio, theta_e_meio, ar)
    else:
        ar_meio = arazao[0:x_index+2]
        theta_n_meio = theta_n[0:x_index+2]
        theta_e_meio = theta_e[0:x_index+2]
        tn_valor = interpolar(ar_meio, theta_n_meio, ar)
        te_valor = interpolar(ar_meio, theta_e_meio, ar)

    return Ln, np.radians(tn_valor), np.radians(te_valor)


def interpolar(x_lista, y_lista, x):
    """
    Interpolação linear simples
    """
    if any(y - x <= 0 for x, y in zip(x_lista, x_lista[1:])):
        raise ValueError("x_lista deve estar em ordem estritamente crescente!")

    intervalos = zip(x_lista, x_lista[1:], y_lista, y_lista[1:])
    inclinacoes = [(y2 - y1) / (x2 - x1) for x1, x2, y1, y2 in intervalos]

    if x <= x_lista[0]:
        return y_lista[0]
    elif x >= x_lista[-1]:
        return y_lista[-1]
    else:
        i = bisect_left(x_lista, x) - 1
        return y_lista[i] + inclinacoes[i] * (x - x_lista[i])


def encontrar_mais_proximo(array, valor):
    """
    Encontra o índice mais próximo na lista para o valor dado
    """
    array = np.asarray(array)
    idx = (np.abs(array - valor)).argmin()
    return idx, array[idx]


def plotar_tubeira(ax, titulo, Rt, angulos, contorno, arazao):
    """
    Plota o contorno da tubeira
    """
    # Ângulos das paredes
    comprimento_tubeira = angulos[0]
    theta_n = angulos[1]
    theta_e = angulos[2]

    # Valores do contorno
    xe = contorno[0]
    ye = contorno[1]
    nye = contorno[2]
    xe2 = contorno[3]
    ye2 = contorno[4]
    nye2 = contorno[5]
    xsino = contorno[6]
    ysino = contorno[7]
    nysino = contorno[8]

    # Configurar aspecto igual
    ax.set_aspect('equal')

    # Plotar seção de entrada da garganta
    ax.plot(xe, ye, linewidth=2.5, color='g', label='Entrada da garganta')
    ax.plot(xe, nye, linewidth=2.5, color='g')

    # Plotar seção de saída da garganta
    ax.plot(xe2, ye2, linewidth=2.5, color='r', label='Saída da garganta')
    ax.plot(xe2, nye2, linewidth=2.5, color='r')

    # Plotar seção do sino
    ax.plot(xsino, ysino, linewidth=2.5, color='b', label='Seção do sino')
    ax.plot(xsino, nysino, linewidth=2.5, color='b')

    # Adicionar dimensões importantes
    # Raio da garganta
    ax.annotate('', xy=(0, Rt), xytext=(0, 0),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(0.1, Rt/2, f'Rt = {Rt}', fontsize=9)

    # Raio de saída
    Re = np.sqrt(arazao) * Rt
    ax.annotate('', xy=(xsino[-1], Re), xytext=(xsino[-1], 0),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(xsino[-1] + 0.1, Re/2, f'Re = {Re:.1f}', fontsize=9)

    # Comprimento da tubeira
    ax.annotate('', xy=(xsino[-1], -Re-10), xytext=(0, -Re-10),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1))
    ax.text(xsino[-1]/2, -Re-15, f'Ln = {comprimento_tubeira:.1f}', fontsize=9, ha='center')

    # Eixos
    ax.axhline(color='black', lw=0.5, linestyle="dashed")
    ax.axvline(color='black', lw=0.5, linestyle="dashed")

    # Grade
    ax.grid(True, alpha=0.3)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', alpha=0.5)
    ax.grid(which='minor', linestyle=':', linewidth='0.5', alpha=0.3)

    # Título e legenda
    ax.set_title(titulo, fontsize=12)
    ax.legend()
    ax.set_xlabel('Posição axial')
    ax.set_ylabel('Raio')


def gerar_tabela_pontos(contorno, nome_arquivo=None):
    # Gera uma tabela com pontos (x, y) para o CAD da tubeira.
    xe, ye, nye, xe2, ye2, nye2, xsino, ysino, nysino = contorno

    # Junta todos os pontos (simetria)
    x_total = np.concatenate([xe, xe2, xsino])
    y_total = np.concatenate([ye, ye2, ysino])

    # Cria DataFrame
    df = pd.DataFrame({
        "x (posição axial)": x_total,
        "y (raio)": y_total
    })

    # Salvar em CSV se solicitado
    if nome_arquivo:
        df.to_csv(nome_arquivo, index=False)

    return df



def plotar_3d(ax, contorno):
    """
    Cria visualização 3D da tubeira
    """
    xe = contorno[0]
    ye = contorno[1]
    xe2 = contorno[3]
    ye2 = contorno[4]
    xsino = contorno[6]
    ysino = contorno[7]

    # Combinar todas as coordenadas
    x = np.concatenate([xe, xe2, xsino])
    y = np.concatenate([ye, ye2, ysino])

    # Criar superfície de revolução
    theta = np.linspace(0, 2*np.pi, 50)
    X, THETA = np.meshgrid(x, theta)
    Y = np.outer(np.cos(theta), y)
    Z = np.outer(np.sin(theta), y)

    # Plotar superfície
    ax.plot_surface(Y, Z, X, alpha=0.8, cmap='viridis')

    # Configurar eixos
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.set_zlabel('X (Posição axial)')
    ax.set_title('Vista 3D da tubeira')


def plotar_completo(titulo, r_garganta, angulos, contorno, arazao):
    """
    Cria plot completo com vista 2D e 3D
    """
    fig = plt.figure(figsize=(15, 7))

    # Plot 2D
    ax1 = fig.add_subplot(121)
    plotar_tubeira(ax1, titulo, r_garganta, angulos, contorno, arazao)

    # Plot 3D
    ax2 = fig.add_subplot(122, projection='3d')
    plotar_3d(ax2, contorno)

    plt.show()