from heston_adi import *


def get_s_mesh(xi, K, c):
    return K + c * np.sinh(xi)


def get_v_mesh(xi, d):
    return d * np.sinh(xi)


def make_grid(m1, m2, S_max, S_0, K, V_max, V_0, s_scale, v_scale):

    # --- get the S-direction mesh and increments
    d_xi = (1 / m1) * (np.arcsinh((S_max - K) / s_scale) - np.arcsinh(-K / s_scale))
    xi_grid = [np.arcsinh(-K / s_scale) + i * d_xi for i in range(m1 + 1)]
    s_grid = [get_s_mesh(xi_grid[i], K, s_scale) for i in range(m1 + 1)]
    s_grid.append(S_0)
    s_grid.sort()
    s_grid.pop(-1)
    delta_s = [s_grid[i + 1] - s_grid[i] for i in range(m1)]

    # --- get the V-direction mesh and increments
    delta_eta = (1 / m2) * np.arcsinh(V_max / v_scale)
    delta_grid = [i * delta_eta for i in range(m2 + 1)]
    v_grid = [get_v_mesh(delta_grid[i], v_scale) for i in range(m2 + 1)]
    v_grid.append(V_0)
    v_grid.sort()
    v_grid.pop(-1)
    delta_v = [v_grid[i + 1] - v_grid[i] for i in range(m2)]

    S_mesh, V_mesh = np.meshgrid(s_grid, v_grid)

    # # --- grid checking
    _ = plt.figure()
    plt.plot(S_mesh, V_mesh, '.', color='blue')
    plt.title("Mesh [0, S] x [0, V]")
    plt.show()

    return s_grid, delta_s, v_grid, delta_v, S_mesh, V_mesh
