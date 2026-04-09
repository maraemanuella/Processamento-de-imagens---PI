def rotulacao_componentes_conectadas(imagem):
    if not imagem or not imagem[0]:
        return []

    rows = len(imagem)
    cols = len(imagem[0])

    labels = [[0] * cols for _ in range(rows)]

    next_label = 1
    parent = {}

    def find(x):
        """Encontra o rótulo raiz (representante) de um conjunto."""
        if x not in parent:
            return x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        """Une dois rótulos (registra que são equivalentes)."""
        px = find(x)
        py = find(y)
        if px != py:
            parent[px] = py

    # 3. PRIMEIRA PASSAGEM - VARREDURA
    for i in range(rows):
        for j in range(cols):

            if imagem[i][j] == 0:
                continue
            
            r = labels[i][j - 1] if j > 0 else 0
            s = labels[i - 1][j] if i > 0 else 0

            if r == 0 and s == 0:
                labels[i][j] = next_label
                parent[next_label] = next_label   # inicializa o conjunto
                next_label += 1

            elif r != 0 and s == 0:
                labels[i][j] = r

            elif r == 0 and s != 0:
                labels[i][j] = s

            else:
                if r == s:
                    
                    labels[i][j] = r
                else:
                    labels[i][j] = r
                    union(r, s)

    # 4. SEGUNDA PASSAGEM - RESOLUÇÃO DE EQUIVALÊNCIAS
    for i in range(rows):
        for j in range(cols):
            if labels[i][j] != 0:
                labels[i][j] = find(labels[i][j])

    # 5. REMAPEAMENTO OPCIONAL
    #    Transforma os rótulos em números consecutivos 1, 2, 3...
    label_map = {}
    current_label = 1
    for i in range(rows):
        for j in range(cols):
            if labels[i][j] != 0:
                old = labels[i][j]
                if old not in label_map:
                    label_map[old] = current_label
                    current_label += 1
                labels[i][j] = label_map[old]

    return labels


# Exercício da página 35
if __name__ == "__main__":
    imagem_exercicio = [
        [0, 0, 0, 0, 0, 255],
        [0, 255, 255, 0, 0, 0],
        [0, 255, 255, 255, 0, 0],
        [0, 0, 255, 255, 0, 0],
        [0, 0, 255, 0, 0, 255],
        [0, 0, 0, 0, 0, 255]
    ]

    rotulos = rotulacao_componentes_conectadas(imagem_exercicio)

    print("Matriz de rótulos:")
    for linha in rotulos:
        print(linha)