from .analytical import (
    displacement_exact_analytical_solution,
    stress_exact_analytical_solution,
)
from .libraries import *

def evaluate_model(model, inputs, x, y, L, h, E, mu):
    prediction = model.predict_fields(inputs)

    exact_sigma = stress_exact_analytical_solution(x=x, y=y, L=L, h=h)
    exact_disp = displacement_exact_analytical_solution(x=x, y=y, L=L, E=E, mu=mu, h=h)

    return prediction, exact_sigma, exact_disp

def compute_stress_errors(x, y, stress_xx, stress_yy, stress_xy, L):


    exact_sigma = stress_exact_analytical_solution(x=x, y=y, L=L)

    exact_sigxx = np.asarray(exact_sigma["exact_sigma_xx"])
    exact_sigyy = np.asarray(exact_sigma["exact_sigma_yy"])
    exact_sigxy = np.asarray(exact_sigma["exact_sigma_xy"])

    stress_xx = np.asarray(stress_xx)
    stress_yy = np.asarray(stress_yy)
    stress_xy = np.asarray(stress_xy)

    error_sigxx = stress_xx - exact_sigxx
    error_sigyy = stress_yy - exact_sigyy
    error_sigxy = stress_xy - exact_sigxy

    return {
        "exact_sigxx": exact_sigxx,
        "exact_sigyy": exact_sigyy,
        "exact_sigxy": exact_sigxy,
        "error_sigxx": error_sigxx,
        "error_sigyy": error_sigyy,
        "error_sigxy": error_sigxy,
    }


def compute_exact_displacement(x, y, L):
    disp_dict = displacement_exact_analytical_solution(x=x, y=y, L=L)

    dx = np.asarray(disp_dict["dispx"])
    dy = np.asarray(disp_dict["dispy"])

    return {
        "exact_disp_x": dx,
        "exact_disp_y": dy,
    }


def compute_relative_stress_errors(
    stress_xx,
    stress_yy,
    stress_xy,
    exact_sigxx,
    exact_sigyy,
    exact_sigxy,
    eps=1e-12,
):
    

    stress_xx = np.asarray(stress_xx, dtype=float)
    stress_yy = np.asarray(stress_yy, dtype=float)
    stress_xy = np.asarray(stress_xy, dtype=float)

    exact_sigxx = np.asarray(exact_sigxx, dtype=float)
    exact_sigyy = np.asarray(exact_sigyy, dtype=float)
    exact_sigxy = np.asarray(exact_sigxy, dtype=float)

    rel_error_sigxx = np.full_like(stress_xx, np.nan, dtype=float)
    rel_error_sigyy = np.full_like(stress_yy, np.nan, dtype=float)
    rel_error_sigxy = np.full_like(stress_xy, np.nan, dtype=float)

    mask_xx = np.abs(exact_sigxx) > eps
    mask_yy = np.abs(exact_sigyy) > eps
    mask_xy = np.abs(exact_sigxy) > eps

    rel_error_sigxx[mask_xx] = (
        stress_xx[mask_xx] - exact_sigxx[mask_xx]
    ) / exact_sigxx[mask_xx]

    rel_error_sigyy[mask_yy] = (
        stress_yy[mask_yy] - exact_sigyy[mask_yy]
    ) / exact_sigyy[mask_yy]

    rel_error_sigxy[mask_xy] = (
        stress_xy[mask_xy] - exact_sigxy[mask_xy]
    ) / exact_sigxy[mask_xy]

    return {
        "rel_error_sigxx": rel_error_sigxx,
        "rel_error_sigyy": rel_error_sigyy,
        "rel_error_sigxy": rel_error_sigxy,
    }