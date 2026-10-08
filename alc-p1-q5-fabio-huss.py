import numpy as np


def resolve_lu(A, b, tol=1e-12):
    """
    Resolve A x = b usando A = L U, sem pivotamento.

    Retorna, nesta ordem:
        L: matriz triangular inferior, com diagonal igual a 1
        U: matriz triangular superior
        x: vetor solução
    """

    # Converte as entradas para arrays numéricos sem alterar as entradas originais.
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    # Validação das dimensões.
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A deve ser uma matriz quadrada.")

    n = A.shape[0]

    if b.ndim != 1 or len(b) != n:
        raise ValueError("b deve ser um vetor com o mesmo número de linhas de A.")

    # A = L U; L começa como identidade e U como matriz nula.
    L = np.eye(n)
    U = np.zeros((n, n))

    # Decomposição LU sem pivotamento.
    for k in range(n):

        # Calcula a linha k de U, da diagonal em diante.
        for j in range(k, n):
            soma = 0.0
            for s in range(k):
                soma += L[k, s] * U[s, j]

            U[k, j] = A[k, j] - soma

        # Sem pivotamento, um pivô nulo impede a continuação.
        if abs(U[k, k]) <= tol:
            raise Exception(
                f"Pivô nulo ou muito próximo de zero na posição ({k}, {k}). "
                "A decomposição LU sem pivotamento não pode ser feita; "
                "use um método com pivotamento ou outra função para resolver o sistema."
            )

        # Calcula os elementos abaixo da diagonal de L.
        for i in range(k + 1, n):
            soma = 0.0
            for s in range(k):
                soma += L[i, s] * U[s, k]

            L[i, k] = (A[i, k] - soma) / U[k, k]

    # Substituição progressiva: resolve L y = b.
    y = np.zeros(n)

    for i in range(n):
        soma = 0.0
        for j in range(i):
            soma += L[i, j] * y[j]

        y[i] = (b[i] - soma) / L[i, i]

    # Substituição regressiva: resolve U x = y.
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        soma = 0.0
        for j in range(i + 1, n):
            soma += U[i, j] * x[j]

        x[i] = (y[i] - soma) / U[i, i]

    return L, U, x