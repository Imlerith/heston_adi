import numpy as np

import heston_adi as hadi


# --- parameters
K = 100.0
S_0 = K
S_max = 8 * K
V_0 = 0.04
V_max = 5.0
T = 1.0

r_d = 0.025
r_f = 0.0
rho = -0.9
sigma = 0.3
kappa = 1.5
eta = 0.04

# --- true price from Monte-Carlo simulation
true_price = 8.8948693600540167

# --- for the spatial grid [0, S] x [0, V]
m1 = 50                                     # S
m2 = 25                                     # V
m = (m1 + 1) * (m2 + 1)                     # matrix A and vector U size
s_scale = K / 5
v_scale = V_max / 1000

# --- for the temporal grid [0, T]
N = 20
delta_t = T / N

# --- explicit-implicit tradeoff parameter
theta = 0.8


# ======================= 1. Set up the model ============================
# --- make the grid
s_grid, delta_s, v_grid, delta_v, s_mesh, v_mesh = hadi.make_grid(m1, m2, S_max, S_0, K, V_max, V_0, s_scale, v_scale)

# --- produce the matrices of discretized derivatives
A_0, A_1, A_2, A = hadi.make_A_matrices(m1, m2, rho, sigma, r_d, r_f, kappa, eta, s_grid, v_grid, delta_s, delta_v)
b_0, b_1, b_2, b = hadi.make_boundaries(m1, m2, r_d, r_f, N, s_grid, delta_t)


# ======================= 2. Perform the pricing ============================
index_s = s_grid.index(S_0)
index_v = v_grid.index(V_0)

# --- (a) Crank-Nicholson scheme
print("\n--- Crank-Nicholson scheme")
UU_0 = np.array([[max(s_grid[i] - K, 0) for i in range(m1 + 1)] for j in range(m2 + 1)])
U_0 = UU_0.flatten()
price_cn, time_cn = hadi.CN_scheme(m, N, U_0, delta_t, A, b, r_f)
price_cn = np.reshape(price_cn, (m2 + 1, m1 + 1))
print(" CN price: ", price_cn[index_v, index_s])
print(" CN error: ", abs(price_cn[index_v, index_s] - true_price) / true_price)
print(" CN computation Time: ", time_cn)

# --- (b) Craig-Sneyd scheme
print("\n--- Craig-Sneyd scheme")
price_cs, time_cs = hadi.CS_scheme(m, N, U_0, delta_t, theta, A, A_0, A_1, A_2, b, b_0, b_1, b_2, r_f)
price_cs = np.reshape(price_cs, (m2 + 1, m1 + 1))
print(" CS price: ", price_cs[index_v, index_s])
print(" CS error: ", abs(price_cs[index_v, index_s] - true_price) / true_price)
print(" CS computation Time: ", time_cs)

# --- (c) Modified Craig-Sneyd scheme
print("\n--- Modified Craig-Sneyd scheme")
price_mcs, time_mcs = hadi.MCS_scheme(m, N, U_0, delta_t, theta, A, A_0, A_1, A_2, b, b_0, b_1, b_2, r_f)
price_mcs = np.reshape(price_mcs, (m2 + 1, m1 + 1))
print(" MCS price: ", price_mcs[index_v, index_s])
print(" MCS error: ", abs(price_mcs[index_v, index_s] - true_price) / true_price)
print(" MCS computation Time: ", time_mcs)

