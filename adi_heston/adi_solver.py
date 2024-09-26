from heston_adi import *


def F(n, omega, A, b, r_f, dt):
    return A * omega + b * np.exp(r_f * dt * n)


def F_0(n, omega, A_0, b_0, r_f, dt):
    return A_0 * omega + b_0 * np.exp(r_f * dt * n)


def F_1(n, omega, A_1, b_1, r_f, dt):
    return A_1 * omega + b_1 * np.exp(r_f * dt * n)


def F_2(n, omega, A_2, b_2, r_f, dt):
    return A_2 * omega + b_2 * np.exp(r_f * dt * n)


def CN_scheme(m, N, U_0, dt, A, b, r_f):
    """
    Crank-Nicholson scheme (not ADI, just for comparison)
    :param m: number of spatial grid points
    :param N: number of temporal grid points
    :param U_0: starting value for the discretized solution
    :param dt: time step for the temporal grid
    :param A: matrix of discretized derivatives
    :param b: vector of boundary values
    :param r_f: risk-free rate
    :return: solution at time N, calc time
    """
    start = datetime.now()
    U = U_0
    I = np.identity(m)
    inv_lhs = inv(csc_matrix(I - 0.5 * dt * A))
    for n in range(1, N + 1):
        U = inv_lhs * (U + 0.5 * dt * (F(n - 1, U, A, b, r_f, dt) + b * np.exp(r_f * dt * n)))
    end = datetime.now()
    time = (end - start).total_seconds()
    return U, time


def CS_scheme(m, N, U_0, dt, theta, A, A_0, A_1, A_2, b, b_0, b_1, b_2, r_f):
    """
    Craig-Sneyd scheme
    :param m: number of spatial grid points
    :param N: number of temporal grid points
    :param U_0: starting value for the discretized solution
    :param dt: time step for the temporal grid
    :param theta: explicit-implicit tradeoff parameter
    :param A: matrix of discretized derivatives
    :param A_0: sub-matrix of discretized derivatives: mixed derivatives
    :param A_1: sub-matrix of discretized derivatives: S-derivatives plus half of the u-term
    :param A_2: sub-matrix of discretized derivatives: V-derivatives plus half of the u-term
    :param b: vector of boundary values
    :param b_0: sub-vector of boundary values
    :param b_1: sub-vector of boundary values
    :param b_2: sub-vector of boundary values
    :param r_f: risk-free rate
    :return: solution at time N, calc time
    """
    start = datetime.now()
    U = U_0
    I = np.identity(m)
    inv_lhs_1 = inv(csc_matrix(I - theta * dt * A_1))
    inv_lhs_2 = inv(csc_matrix(I - theta * dt * A_2))
    for n in range(0, N):
        Y_0 = U + dt * F(n - 1, U, A, b, r_f, dt)
        rhs_1 = Y_0 + theta * dt * (b_1 * np.exp(r_f * dt * n) - F_1(n - 1, U, A_1, b_1, r_f, dt))
        Y_1 = inv_lhs_1 * rhs_1
        rhs_2 = Y_1 + theta * dt * (b_2 * np.exp(r_f * dt * n) - F_2(n - 1, U, A_2, b_2, r_f, dt))
        Y_2 = inv_lhs_2 * rhs_2

        Y_0_tilde = Y_0 + 0.5 * dt * (F_0(n, Y_2, A_0, b_0, r_f, dt) - F_0(n - 1, U, A_0, b_0, r_f, dt))
        rhs_1 = Y_0_tilde + theta * dt * (b_1 * np.exp(r_f * dt * n) - F_1(n - 1, U, A_1, b_1, r_f, dt))
        Y_1_tilde = inv_lhs_1 * rhs_1
        rhs_2 = Y_1_tilde + theta * dt * (b_2 * np.exp(r_f * dt * n) - F_2(n - 1, U, A_2, b_2, r_f, dt))
        U = inv_lhs_2 * rhs_2
    end = datetime.now()
    time = (end - start).total_seconds()
    return U, time


def MCS_scheme(m, N, U_0, delta_t, theta, A, A_0, A_1, A_2, b, b_0, b_1, b_2, r_f):
    """
    Modified Craig-Sneyd scheme
    :param m:
    :param N:
    :param U_0:
    :param delta_t:
    :param theta:
    :param A:
    :param A_0:
    :param A_1:
    :param A_2:
    :param b:
    :param b_0:
    :param b_1:
    :param b_2:
    :param r_f:
    :return:
    """
    start = datetime.now()
    U = U_0
    I = np.identity(m)
    inv_lhs_1 = inv(csc_matrix(I - theta * delta_t * A_1))
    inv_lhs_2 = inv(csc_matrix(I - theta * delta_t * A_2))
    for n in range(0, N):
        Y_0 = U + delta_t * F(n - 1, U, A, b, r_f, delta_t)
        rhs_1 = Y_0 + theta * delta_t * (b_1 * np.exp(r_f * delta_t * n) - F_1(n - 1, U, A_1, b_1, r_f, delta_t))
        Y_1 = inv_lhs_1 * rhs_1
        rhs_2 = Y_1 + theta * delta_t * (b_2 * np.exp(r_f * delta_t * n) - F_2(n - 1, U, A_2, b_2, r_f, delta_t))
        Y_2 = inv_lhs_2 * rhs_2

        Y_0_hat = Y_0 + theta * delta_t * (F_0(n, Y_2, A_0, b_0, r_f, delta_t) - F_0(n - 1, U, A_0, b_0, r_f, delta_t))
        Y_0_tilde = Y_0_hat + (0.5 - theta) * delta_t * (F(n, Y_2, A, b, r_f, delta_t) - F(n - 1, U, A, b, r_f, delta_t))

        rhs_1 = Y_0_tilde + theta * delta_t * (b_1 * np.exp(r_f * delta_t * n) - F_1(n - 1, U, A_1, b_1, r_f, delta_t))
        Y_1_tilde = inv_lhs_1 * rhs_1
        rhs_2 = Y_1_tilde + theta * delta_t * (b_2 * np.exp(r_f * delta_t * n) - F_2(n - 1, U, A_2, b_2, r_f, delta_t))
        U = inv_lhs_2 * rhs_2
    end = datetime.now()
    time = (end - start).total_seconds()
    return U, time

