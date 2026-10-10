import torch


def viterbi(x: torch.Tensor, transitions: torch.Tensor):
    """
    x: (T, C) emission log probabilities
    transitions: (C, C) log transition matrix, [prev, curr]

    returns:
        path: (T,) most likely state sequence
    """
    T, C = x.shape
    device = x.device

    backpointer = torch.zeros((T, C), dtype=torch.long, device=device)

    # init
    V_prev = x[0, :]

    # recursion
    for t in range(1, T):
        # (C_prev, C_curr)
        scores = V_prev[:, None] + transitions  # shape (C, C)

        best_scores, best_states = torch.max(scores, dim=0)

        V_curr = x[t, :] + best_scores
        V_prev = V_curr

        backpointer[t, :] = best_states

    # backtracking
    path = torch.zeros(T, dtype=torch.long, device=device)

    # last state
    path[T - 1] = torch.argmax(V_prev)

    for t in reversed(range(T - 1)):
        path[t] = backpointer[t + 1, path[t + 1]]

    return path
