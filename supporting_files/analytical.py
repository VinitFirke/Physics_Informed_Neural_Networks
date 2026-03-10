from .libraries import *
def displacement_exact_analytical_solution(x, y, L, F=0.01, E=1.0, mu=0.3, h=1.0):
    I = (h**3) / 12
    D_1 = (F * y) / (6 * E * I)

    dispx = D_1 * (((6 * L - 3 * x) * x) + ((2 + mu) * y**2) - (1.5 * h**2 * (1 + mu)))
    dispy = -D_1 * ((3 * mu * y**2 * (L - x)) + ((3 * L - x) * x**2))

    return {"dispx": dispx, "dispy": dispy}


def stress_exact_analytical_solution(x, y, L, F=0.01, h=1.0):
    C_1 = (6 * F) / (h**3)

    stress_exact_xx = C_1 * 2 * x * y
    stress_exact_yy = np.zeros_like(stress_exact_xx)
    stress_exact_xy = C_1 * (((h / 2) ** 2) - y**2)

    return {
        "exact_sigma_xx": stress_exact_xx,
        "exact_sigma_yy": stress_exact_yy,
        "exact_sigma_xy": stress_exact_xy,
    }