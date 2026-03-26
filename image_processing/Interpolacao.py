def reduzir_vizinho_proximo(img):

    linhas = len(img)
    colunas = len(img[0])

    nova = []

    for i in range(0, linhas, 2):
        linha = []
        for j in range(0, colunas, 2):
            linha.append(img[i][j])
        nova.append(linha)

    return nova

def ampliar_vizinho_proximo(img):

    linhas = len(img)
    colunas = len(img[0])

    nova_linhas = linhas * 2 - 1
    nova_colunas = colunas * 2 - 1

    nova = [[0 for _ in range(nova_colunas)] for _ in range(nova_linhas)]

    # copiar pixels originais
    for i in range(linhas):
        for j in range(colunas):
            nova[i*2][j*2] = img[i][j]

    for i in range(0, nova_linhas-2, 2):
        for j in range(0, nova_colunas-2, 2):

            f_ij = nova[i][j]
            f_ij1 = nova[i][j+2]

            # pixel superior
            nova[i][j+1] = f_ij

            # pixel esquerda
            nova[i+1][j] = f_ij

            # pixel central
            nova[i+1][j+1] = f_ij

            # pixel direita
            nova[i+1][j+2] = f_ij1

            # pixel inferior
            nova[i+2][j+1] = nova[i+2][j]

    return nova

def reduzir_bilinear(img):

    linhas = len(img)
    colunas = len(img[0])

    nova_linhas = linhas // 2
    nova_colunas = colunas // 2

    nova = [[0 for _ in range(nova_colunas)] for _ in range(nova_linhas)]

    for i in range(nova_linhas):
        for j in range(nova_colunas):

            p1 = img[2*i][2*j]
            p2 = img[2*i][2*j+1]
            p3 = img[2*i+1][2*j]
            p4 = img[2*i+1][2*j+1]

            media = (p1 + p2 + p3 + p4) / 4
            nova[i][j] = round(media)

    return nova


def ampliar_bilinear(img):

    linhas = len(img)
    colunas = len(img[0])

    nova_linhas = linhas * 2 - 1
    nova_colunas = colunas * 2 - 1

    nova = [[0 for _ in range(nova_colunas)] for _ in range(nova_linhas)]

    # copiar pixels originais
    for i in range(linhas):
        for j in range(colunas):
            nova[i*2][j*2] = img[i][j]

    for i in range(0, nova_linhas-2, 2):
        for j in range(0, nova_colunas-2, 2):

            f_ij = nova[i][j]
            f_ij1 = nova[i][j+2]
            f_i1j = nova[i+2][j]
            f_i1j1 = nova[i+2][j+2]

            nova[i][j+1] = round((f_ij + f_ij1) / 2)

            nova[i+1][j] = round((f_ij + f_i1j) / 2)

            nova[i+1][j+1] = round((f_ij + f_ij1 + f_i1j + f_i1j1) / 4)

            nova[i+1][j+2] = round((f_ij1 + f_i1j1) / 2)

            nova[i+2][j+1] = round((f_i1j + f_i1j1) / 2)

    return nova


def imprimir(matriz):
    for linha in matriz:
        print(linha)
    print()


imagem = [
    [20,40,40,20],
    [40,20,54,30],
    [60,30,80,40],
    [70,70,20,60]
]

print("Imagem original")
imprimir(imagem)

print("Redução vizinho mais próximo")
imprimir(reduzir_vizinho_proximo(imagem))

print("Ampliação vizinho mais próximo")
imprimir(ampliar_vizinho_proximo(imagem))

print("Redução bilinear")
imprimir(reduzir_bilinear(imagem))

print("Ampliação bilinear")
imprimir(ampliar_bilinear(imagem))