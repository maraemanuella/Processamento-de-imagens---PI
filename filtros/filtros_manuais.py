"""
Processamento Digital de Imagens
Aula 9 e Aula 10 - Implementações Manuais
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def load_image(image_path):
    print(f"Tentando carregar: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Arquivo não encontrado: {image_path}")
        print("→ Usando imagem sintética de teste...\n")
        img = np.zeros((400, 400), dtype=np.uint8)
        img[80:320, 80:320] = 180
        cv2.rectangle(img, (120, 120), (280, 280), 80, -1)
        cv2.line(img, (100, 100), (300, 300), 255, 8)
        return img
    
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Falha ao carregar como grayscale. Tentando como colorida...")
        img_color = cv2.imread(image_path)
        if img_color is not None:
            img = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
            print("Imagem colorida convertida para tons de cinza.")
        else:
            print("Não foi possível carregar a imagem. Usando teste.")
            img = np.zeros((400, 400), dtype=np.uint8)
            img[80:320, 80:320] = 180
    
    print(f"Imagem carregada | Dimensões: {img.shape}\n")
    return img


# ====================== 1. FILTRO DA MÉDIA ======================
def mean_filter_manual(img, kernel_size=3, padding_type='zero'):
    """Filtro da Média - Implementação (Aula 9)"""
    img = img.astype(np.float32)
    h, w = img.shape
    k = kernel_size
    pad = k // 2

    if padding_type == 'zero':
        padded = np.zeros((h + 2*pad, w + 2*pad), dtype=np.float32)
        padded[pad:pad+h, pad:pad+w] = img
    else:  # replicate
        padded = np.pad(img, pad, mode='edge')

    output = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            output[i, j] = np.mean(padded[i:i+k, j:j+k])
    
    return np.clip(output, 0, 255).astype(np.uint8)


# ====================== 2. FILTRO LAPLACIANO ======================
def laplacian_manual(img, mask_type=1):
    """Implementa as 4 máscaras de Laplaciano dos slides (Aula 10)"""
    img = img.astype(np.float32)
    h, w = img.shape
    pad = 1
    
    padded = np.zeros((h + 2, w + 2), dtype=np.float32)
    padded[1:1+h, 1:1+w] = img

    kernels = {
        1: np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),   # centro negativo
        2: np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]),
        3: np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]), # centro positivo
        4: np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    }
    
    kernel = kernels.get(mask_type)
    if kernel is None:
        raise ValueError("mask_type deve ser 1, 2, 3 ou 4")

    output = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            output[i, j] = np.sum(padded[i:i+3, j:j+3] * kernel)
    
    return output


def show_laplacians_comparison(img):
    print("Gerando comparação dos filtros Laplacianos...")
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagem Original')
    plt.axis('off')
    
    for i in range(1, 5):
        lap = laplacian_manual(img, i)
        lap_norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        plt.subplot(2, 3, i + 1)
        plt.imshow(lap_norm, cmap='gray')
        plt.title(f'Laplaciano - Máscara {i}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# ====================== 3. FILTRO SOBEL ======================
def sobel_manual(img):
    img = img.astype(np.float32)
    h, w = img.shape
    pad = 1
    padded = np.zeros((h + 2, w + 2), dtype=np.float32)
    padded[1:1+h, 1:1+w] = img

    Gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Gx = np.zeros((h, w), dtype=np.float32)
    Gy = np.zeros((h, w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            window = padded[i:i+3, j:j+3]
            Gx[i, j] = np.sum(window * Gx_kernel)
            Gy[i, j] = np.sum(window * Gy_kernel)
    
    return Gx, Gy


def process_sobel(img):
    Gx, Gy = sobel_manual(img)
    mag = np.sqrt(Gx**2 + Gy**2)
    
    # (a) Clipping: valores negativos → 0
    mag_clipped = np.clip(mag, 0, 255).astype(np.uint8)
    
    # (b) Normalização para [0, 255]
    mag_normalized = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    plt.figure(figsize=(14, 9))
    titles = ['Original', 'Gx', 'Gy', 
              'Magnitude (Clipped)', 
              'Magnitude (Normalizada)', 
              'Aproximação |Gx| + |Gy|']
    
    images = [img, Gx, Gy, mag_clipped, mag_normalized, np.abs(Gx) + np.abs(Gy)]
    
    for idx, (title, image) in enumerate(zip(titles, images)):
        plt.subplot(2, 3, idx + 1)
        if idx in [1, 2]:  # Gx e Gy
            plt.imshow(image, cmap='gray', vmin=-400, vmax=400)
        else:
            plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return mag_normalized


# ====================== MAIN ======================
def main():

    image_path = r'/home/fabrica/Área de Trabalho/operacoes-PI/elefante.jpg'  
    
    img = load_image(image_path)
    
    print("="*70)
    print("PROCESSAMENTO DIGITAL DE IMAGENS - IMPLEMENTAÇÕES MANUAIS")
    print("Aulas 9 e 10 - Versão Final Otimizada")
    print("="*70 + "\n")
    
    mean_filtered = mean_filter_manual(img, kernel_size=3, padding_type='zero')
    print("1. Filtro da Média (3x3) aplicado com Zero Padding")
    
    show_laplacians_comparison(img)
    
    print("3. Aplicando Filtro Sobel...")
    sobel_mag = process_sobel(img)
    
    lap1 = laplacian_manual(img, 1)
    enhanced = np.clip(img.astype(np.float32) - lap1, 0, 255).astype(np.uint8)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1); plt.imshow(img, cmap='gray');           plt.title('Original')
    plt.subplot(2, 3, 2); plt.imshow(mean_filtered, cmap='gray'); plt.title('Filtro da Média')
    plt.subplot(2, 3, 3); plt.imshow(enhanced, cmap='gray');      plt.title('Realce com Laplaciano')
    plt.subplot(2, 3, 4); plt.imshow(sobel_mag, cmap='gray');     plt.title('Sobel - Magnitude')
    plt.tight_layout()
    plt.show()
    
    print("\nPROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("Versão final otimizada: redundâncias removidas.")


if __name__ == "__main__":
    main()
    
