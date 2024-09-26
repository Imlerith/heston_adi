from heston_adi import *


def make_A_matrices(m1, m2, rho, sigma, r_d, r_f, kappa, eta, val_s, val_v, delta_s, delta_v):
    m = (m1 + 1) * (m2 + 1)
    A_0 = np.zeros((m, m))
    A_1 = np.zeros((m, m))
    A_2 = np.zeros((m, m))

    pos_upwind_1_first_d = [-2, -1, 0]
    pos_upwind_2_first_d = [0, 1, 2]
    pos_central_second_d = [-1, 0, 1]
    pos_central_mixed_d = [[-1, 0, 1], [-1, 0, 1]]

    # --- matrix A_0: mixed derivatives
    for j in range(1, m2):
        for i in range(1, m1):
            c = rho * sigma * val_s[i] * val_v[j]
            for k in pos_central_mixed_d[0]:
                for l in pos_central_mixed_d[1]:
                    A_0[i + j * (m1 + 1), i + k + (j + l) * (m1 + 1)] += c * beta_coeff(i - 1, k, delta_s) * beta_coeff(j - 1, l, delta_v)

    A_0 = csc_matrix(A_0)

    _ = plt.figure()
    plt.spy(A_0)
    plt.title("Matrix A_0: mixed derivatives")
    plt.show()

    # --- matrix A_1: S-derivatives plus half of the u-term
    for j in range(m2 + 1):
        for i in range(1, m1):
            a = 0.5 * val_s[i] ** 2 * val_v[j]
            b = (r_d - r_f) * val_s[i]
            for k in pos_central_second_d:
                A_1[i + j * (m1 + 1), i + k + j * (m1 + 1)] += a * delta_coeff(i - 1, k, delta_s) + b * beta_coeff(i - 1, k, delta_s)
            A_1[i + j * (m1 + 1), i + j * (m1 + 1)] += -0.5 * r_d
        A_1[m1 + j * (m1 + 1), m1 + j * (m1 + 1)] += -0.5 * r_d

    A_1 = csc_matrix(A_1)

    _ = plt.figure()
    plt.spy(A_1)
    plt.title("Matrix A_1: S-derivatives plus half of the u-term")
    plt.show()

    # --- matrix A_2: V-derivatives plus half of the u-term
    for j in range(m2 - 1):
        for i in range(m1 + 1):
            temp = kappa * (eta - val_v[j])
            temp2 = 0.5 * sigma ** 2 * val_v[j]
            # --- if V > 1, apply one of the "upwind" schemes for first derivative approximation
            if val_v[j] > 1.0:
                for k in pos_upwind_1_first_d:
                    A_2[i + (j + 1) * (m1 + 1), i + (m1 + 1) * (j + 1 + k)] += temp * alpha_coeff(j, k, delta_v)
                for k in pos_central_second_d:
                    A_2[i + (j + 1) * (m1 + 1), i + (m1 + 1) * (j + 1 + k)] += temp2 * delta_coeff(j - 1, k, delta_v)
            # --- apply the second "upwind" scheme at the boundary V=0
            if j == 0:
                for k in pos_upwind_2_first_d:
                    A_2[i, i + (m1 + 1) * k] += temp * gamma_coeff(j, k, delta_v)
            else:
                for k in pos_central_second_d:
                    A_2[i + j * (m1 + 1), i + (m1 + 1) * (j + k)] += temp * beta_coeff(j - 1, k, delta_v) + temp2 * delta_coeff(j - 1, k, delta_v)
            A_2[i + j * (m1 + 1), i + j * (m1 + 1)] += -0.5 * r_d

    A_2 = csc_matrix(A_2)

    _ = plt.figure()
    plt.spy(A_2)
    plt.title("Matrix A_2: V-derivatives plus half of the u-term")
    plt.show()

    # --- sum up A_0, A_1, A_2 to get the complete A matrix
    A = A_0 + A_1 + A_2
    A = csc_matrix(A)

    _ = plt.figure()
    plt.spy(A)
    plt.title("Matrix A: all derivatives and the complete u-term")
    plt.show()

    return A_0, A_1, A_2, A


def make_boundaries(m1, m2, r_d, r_f, N, val_s, dt):
    m = (m1 + 1) * (m2 + 1)
    b_0 = [0.0] * m
    b_1 = [0.0] * m
    b_2 = [0.0] * m

    # --- boundary when s = S (each t = N * dt)
    for j in range(m2 + 1):
        b_1[m1 * (j + 1)] = (r_d - r_f) * val_s[-1] * np.exp(-r_f * dt * (N - 1))

    # --- boundary when v = V (each t = N * dt)
    for i in range(1, m1 + 1):
        b_2[m - m1 - 1 + i] = -0.5 * r_d * val_s[i] * np.exp(-r_f * dt * (N - 1))

    # --- get the complete b vector
    b_0 = np.array(b_0)
    b_1 = np.array(b_1)
    b_2 = np.array(b_2)

    b = b_0 + b_1 + b_2

    return b_0, b_1, b_2, b


