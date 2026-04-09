"""
Processameno de Imagens - CIência da Computação - UFT
Profa. Dra. Glenda Botelho
ALUNOS: Mara Emanuella Carvalho Martins e Luís Gustavo Alves Bezerra

Operações implementadas manualmente:
    1. Adição de imagens
    2. Subtração de imagens (com normalização)
    3. Rotação (mapeamento inverso)
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
    """
    Rotaciona a imagem usando mapeamento inverso (Eq. 4.5 do slide).
    Interpolação: vizinho mais próximo.
    """
    altura, largura = f.shape[:2]
    theta = math.radians(angulo_graus)
    cx, cy = largura / 2.0, altura / 2.0

    cos_i = math.cos(-theta)
    sen_i = math.sin(-theta)

    # Matriz de Coordenadas em todos os pixels
    y, x = np.indices((altura, largura))

    # Translada
    xd_c = x - cx
    yd_c = y - cy

    x0 = (cos_i * xd_c - sen_i * yd_c + cx).round().astype(int)
    y0 = (sen_i * xd_c + cos_i * yd_c + cy).round().astype(int)

    destino = np.zeros_like(f)

    # Mascara de Pixels
    mask = (0 <= x0) & (x0 < largura) & (0 <= y0) & (y0 < altura)

    destino[y[mask], x[mask]] = f[y0[mask], x0[mask]]

    return destino


def exibir_e_salvar(imagens: list, titulos: list, nome_arquivo: str):
    """Exibe uma grade de imagens e salva em disco."""
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
    print(f"  → Salvo em: {caminho}")

def demo_aritmetica():
    """Demonstração com os valores do slide (p.18/19)."""
    print("\n" + "="*60)
    print("  OPERAÇÕES ARITMÉTICAS — valores do slide")
    print("="*60)

    g1 = np.array([
        [50, 50, 53, 53, 53],
        [53, 53, 53, 53, 53],
        [53, 53, 46, 46, 46],
        [46, 46, 48, 48, 48],
        [48, 48, 48, 70, 70],
    ], dtype=np.uint8)

    g2 = np.array([
        [120, 120, 113, 113, 114],
        [113, 113, 113,  64,  64],
        [ 64,  64,  64,  64,  64],
        [ 64,  64,  41,  41,  41],
        [ 41,  41,  41,  41,  41],
    ], dtype=np.uint8)

    soma  = adicao_imagens(g1, g2)
    media = (soma.astype(np.int32) // 2).astype(np.uint8)
    diff  = subtracao_imagens(g1, g2)

    print("g1:\n", g1)
    print("\ng2:\n", g2)
    print("\nMédia (g1 + g2)/2:\n", media)
    print("\nSubtração (escalada):\n", diff)

    exibir_e_salvar(
        [g1, g2, media, diff],
        ["g1", "g2", "Adição\n(média)", "Subtração\n(escalada)"],
        "aritmetica_slide.png"
    )

def demo_rotacao():
    """Rotação em múltiplos ângulos com imagem sintética."""
    print("\n" + "="*60)
    print("  ROTAÇÃO — múltiplos ângulos")
    print("="*60)

    tam = 120
    m = tam // 4
    f = np.zeros((tam, tam), dtype=np.uint8)
    f[m:tam-m, m:tam-m] = 180
    f[tam//2, m:tam-m] = 255
    f[m:tam-m, tam//2] = 255

    angulos = [0, 45, 90, 135, 180]
    resultados = []
    titulos = []

    for ang in angulos:
        print(f"  Rotacionando {ang}°...")
        resultados.append(rotacao_imagem(f, ang))
        titulos.append(f"θ = {ang}°")

    exibir_e_salvar(resultados, titulos, "rotacao_angulos.png")

def demo_imagem_real(caminho: str):
    """Aplica as operações em uma imagem real."""
    print("\n" + "="*60)
    print(f"  IMAGEM REAL: {caminho}")
    print("="*60)

    img = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [ERRO] Não foi possível abrir '{caminho}'.")
        return

    g = img & 0b11111110

    print(f"  Dimensões: {img.shape}")

    soma = adicao_imagens(img, g)
    diff = subtracao_imagens(img, g)
    rot  = rotacao_imagem(img, 45)

    exibir_e_salvar(
        [img, g, soma, diff, rot],
        ["f — original", "g — LSB zerado",
         "s = f + g", "d = f − g\n(escalada)", "Rotação 45°"],
        "operacoes_real.png"
    )


if __name__ == "__main__":
    print("\n__________________________________________________")
    print("PDI · UFT — Operações Aritméticas e Geométricas")
    print("__________________________________________________")

    demo_aritmetica()
    demo_rotacao()

    if len(sys.argv) > 1:
        demo_imagem_real(sys.argv[1])