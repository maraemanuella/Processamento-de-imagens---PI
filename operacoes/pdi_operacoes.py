"""
Processamento de Imagens - Ciência da Computação - UFT
Profa. Dra. Glenda Botelho
ALUNOS: Mara Emanuella Carvalho Martins e Luís Gustavo Alves Bezerra

Operações implementadas manualmente:
    1. Adição de imagens
    2. Subtração de imagens (com normalização)
    3. Rotação (mapeamento inverso)
    4. Transformação Power-Law
    5. Equalização de Histograma
"""

import os
import sys
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def adicao_imagens(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    if f.shape != g.shape:
        raise ValueError(f"Tamanhos diferentes: f={f.shape}, g={g.shape}")

    s = f.astype(np.int32) + g.astype(np.int32)
    return np.clip(s, 0, 255).astype(np.uint8)


def subtracao_imagens(f: np.ndarray, g: np.ndarray) -> np.ndarray:
    if f.shape != g.shape:
        raise ValueError(f"Tamanhos diferentes: f={f.shape}, g={g.shape}")

    d = f.astype(np.int32) - g.astype(np.int32)
    d_min, d_max = d.min(), d.max()

    if d_min == d_max:
        return np.zeros_like(f, dtype=np.uint8)

    return ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)


def rotacao_imagem(f: np.ndarray, angulo_graus: float) -> np.ndarray:
    altura, largura = f.shape[:2]
    theta = math.radians(angulo_graus)
    cx, cy = largura / 2.0, altura / 2.0

    cos_i = math.cos(-theta)
    sen_i = math.sin(-theta)

    y, x = np.indices((altura, largura))

    xd_c = x - cx
    yd_c = y - cy

    x0 = (cos_i * xd_c - sen_i * yd_c + cx).round().astype(int)
    y0 = (sen_i * xd_c + cos_i * yd_c + cy).round().astype(int)

    destino = np.zeros_like(f)

    mask = (0 <= x0) & (x0 < largura) & (0 <= y0) & (y0 < altura)

    destino[y[mask], x[mask]] = f[y0[mask], x0[mask]]

    return destino


def power_law_transformacao(f: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    altura, largura = f.shape[:2]
    destino = np.zeros_like(f, dtype=np.uint8)

    for i in range(altura):
        for j in range(largura):
            r = f[i, j] / 255.0
            s = c * (r ** gamma) * 255
            destino[i, j] = np.clip(int(s), 0, 255)

    return destino


def equalizacao_histograma(f: np.ndarray) -> np.ndarray:
    altura, largura = f.shape[:2]
    total_pixels = altura * largura
    L = 256

    hist = np.zeros(L, dtype=np.int32)

    # Histograma manual
    for i in range(altura):
        for j in range(largura):
            intensidade = f[i, j]
            hist[intensidade] += 1

    # CDF manual
    cdf = np.zeros(L, dtype=np.float64)
    acumulado = 0

    for k in range(L):
        acumulado += hist[k]
        cdf[k] = acumulado / total_pixels

    # Mapeamento
    mapa = np.zeros(L, dtype=np.uint8)

    for k in range(L):
        mapa[k] = int((L - 1) * cdf[k])

    # Aplicação manual
    resultado = np.zeros_like(f, dtype=np.uint8)

    for i in range(altura):
        for j in range(largura):
            resultado[i, j] = mapa[f[i, j]]

    return resultado


def exibir_e_salvar(imagens: list, titulos: list, nome_arquivo: str):
    os.makedirs("outputs", exist_ok=True)
    caminho = os.path.join("outputs", nome_arquivo)

    n = len(imagens)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), squeeze=False)
    fig.patch.set_facecolor('#1a1a2e')

    for ax, img, titulo in zip(axes.flat, imagens, titulos):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(titulo, color='white', fontsize=11, pad=8)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(caminho, dpi=130, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"→ Salvo em: {caminho}")


def demo_imagem_real(caminho: str):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[ERRO] Não foi possível abrir '{caminho}'.")
        return

    g = img & 0b11111110
    soma = adicao_imagens(img, g)
    diff = subtracao_imagens(img, g)
    rot = rotacao_imagem(img, 45)

    exibir_e_salvar(
        [img, g, soma, diff, rot],
        ["Original", "LSB zerado", "Adição", "Subtração", "Rotação 45°"],
        "operacoes_real.png"
    )


def demo_transformacao(caminho: str, gamma_valor: float):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[ERRO] Não foi possível abrir '{caminho}'.")
        return

    resultado = power_law_transformacao(img, gamma_valor)

    exibir_e_salvar(
        [resultado],
        [f"Power-Law γ={gamma_valor}"],
        "power_law.png"
    )


def demo_equalizacao(caminho: str):
    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[ERRO] Não foi possível abrir '{caminho}'.")
        return

    resultado = equalizacao_histograma(img)

    exibir_e_salvar(
        [img, resultado],
        ["Original", "Equalizada"],
        "equalizacao_histograma.png"
    )


if __name__ == "__main__":
    print("\n__________________________________________________")
    print("PDI · UFT — Operações Aritméticas e Geométricas")
    print("__________________________________________________")

    if len(sys.argv) > 2:
        comando = sys.argv[1]
        imagem = sys.argv[2]

        if comando == "-r":
            demo_imagem_real(imagem)

        elif comando == "-t":
            gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
            demo_transformacao(imagem, gamma)

        elif comando == "-e":
            demo_equalizacao(imagem)

        else:
            print("Comando inválido.")
            print("Use:")
            print("python script.py -r imagem.png")
            print("python script.py -t imagem.png 0.5")
            print("python script.py -e imagem.png")