import numpy as np

def generate_rectangle_points(L, h, n_inner=2500, n_edge=250):
    x_inner = np.random.uniform(0.0, L, n_inner)
    y_inner = np.random.uniform(-h / 2, h / 2, n_inner)

    x_top = np.random.uniform(0, L, n_edge)
    y_top = np.ones(n_edge) * h / 2

    x_bottom = np.random.uniform(0, L, n_edge)
    y_bottom = np.ones(n_edge) * -h / 2

    x_left = np.zeros(n_edge)
    y_left = np.random.uniform(-h / 2, h / 2, n_edge)

    x_right = np.ones(n_edge) * L
    y_right = np.random.uniform(-h / 2, h / 2, n_edge)

    return {
        "x_inner": x_inner, "y_inner": y_inner,
        "x_top": x_top, "y_top": y_top,
        "x_bottom": x_bottom, "y_bottom": y_bottom,
        "x_left": x_left, "y_left": y_left,
        "x_right": x_right, "y_right": y_right,
    }