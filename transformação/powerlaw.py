import numpy as np

imagem = np.array([
    [52, 55, 61],
    [68, 70, 75],
    [80, 85, 90]
], dtype=np.float32)

gamma = 0.5

linhas = imagem.shape[0]
colunas = imagem.shape[1]

resultado = np.zeros((linhas, colunas), dtype=np.uint8)

for i in range(linhas):
    for j in range(colunas):
        r = imagem[i][j] / 255.0
        s = 255 * (r ** gamma)
        resultado[i][j] = int(s)

print(resultado)