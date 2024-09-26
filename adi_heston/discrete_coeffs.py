def alpha_coeff(i, pos, delta_x):
    if pos == -2:
        return delta_x[i] / (delta_x[i - 1] * (delta_x[i - 1] + delta_x[i]))
    elif pos == -1:
        return (-delta_x[i - 1] - delta_x[i]) / (delta_x[i - 1] * delta_x[i])
    elif pos == 0:
        return (delta_x[i - 1] + 2 * delta_x[i]) / (delta_x[i] * (delta_x[i - 1] + delta_x[i]))
    else:
        raise ValueError("Wrong pos")


def beta_coeff(i, pos, delta_x):
    if pos == -1:
        return -delta_x[i + 1] / (delta_x[i] * (delta_x[i] + delta_x[i + 1]))
    elif pos == 0:
        return (delta_x[i + 1] - delta_x[i]) / (delta_x[i] * delta_x[i + 1])
    elif pos == 1:
        return delta_x[i] / (delta_x[i + 1] * (delta_x[i] + delta_x[i + 1]))
    else:
        raise ValueError("Wrong pos")


def gamma_coeff(i, pos, delta_x):
    if pos == 0:
        return (-2 * delta_x[i + 1] - delta_x[i + 2]) / (delta_x[i + 1] * (delta_x[i + 1] + delta_x[i + 2]))
    elif pos == 1:
        return (delta_x[i + 1] + delta_x[i + 2]) / (delta_x[i + 1] * delta_x[i + 2])
    elif pos == 2:
        return -delta_x[i + 1] / (delta_x[i + 2] * (delta_x[i + 1] + delta_x[i + 2]))
    else:
        raise ValueError("Wrong pos")


def delta_coeff(i, pos, delta_x):
    if pos == -1:
        return 2 / (delta_x[i] * (delta_x[i] + delta_x[i + 1]))
    elif pos == 0:
        return -2 / (delta_x[i] * delta_x[i + 1])
    elif pos == 1:
        return 2 / (delta_x[i + 1] * (delta_x[i] + delta_x[i + 1]))
    else:
        raise ValueError("Wrong pos")

