from supporting_files.libraries import *



import os
import time

from supporting_files.lbfgs import function_factory
from supporting_files.geometry import generate_rectangle_points
from supporting_files.pinn_model import PINN
from supporting_files.trainer import train_with_adam, train_with_lbfgs
from supporting_files.evaluation import (
    compute_stress_errors,
    compute_exact_displacement,
    compute_relative_stress_errors,
    evaluate_model
)
from supporting_files.plotting import (
    plot_loss,
    plot_pinn_stress,
    plot_pinn_displacement,
    plot_updated_points,
    plot_absolute_pinn_vs_exact_stress_error,
    plot_relative_stress_error,
    plot_analytical_displacement
)


def normalize_geometry_output(points_data):
    """
    Supports both:
    1. dict output from geometry.py
    2. tuple/list output from your original function
    """
    if isinstance(points_data, dict):
        return points_data

    if isinstance(points_data, (tuple, list)) and len(points_data) == 10:
        (
            x_inner,
            y_inner,
            x_top,
            y_top,
            x_bottom,
            y_bottom,
            x_left,
            y_left,
            x_right,
            y_right,
        ) = points_data

        return {
            "x_inner": x_inner,
            "y_inner": y_inner,
            "x_top": x_top,
            "y_top": y_top,
            "x_bottom": x_bottom,
            "y_bottom": y_bottom,
            "x_left": x_left,
            "y_left": y_left,
            "x_right": x_right,
            "y_right": y_right,
        }

    raise ValueError(
        "Unsupported output format from generate_rectangle_points(). "
        "Expected dict or tuple/list of length 10."
    )


def main():
    start_time = time.time()

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    tf.random.set_seed(42)
    np.random.seed(42)

    # ------------------------------------------------------------------
    # Problem parameters
    # ------------------------------------------------------------------
    L = 1.0
    h = 1.0
    E = 1.0
    mu = 0.3
    traction_y = 0.01

    # ------------------------------------------------------------------
    # Sampling / training parameters
    # ------------------------------------------------------------------
    n_inner = 2500
    n_edge = 250

    adam_lr = 1e-2
    adam_epochs = 5000
    lbfgs_max_iterations = 20000

    # ------------------------------------------------------------------
    # Output folder
    # ------------------------------------------------------------------
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Geometry / collocation points
    # ------------------------------------------------------------------
    points_data_raw = generate_rectangle_points(L, h, n_inner=n_inner, n_edge=n_edge)
    pts = normalize_geometry_output(points_data_raw)

    x_inner = np.asarray(pts["x_inner"], dtype=np.float32)
    y_inner = np.asarray(pts["y_inner"], dtype=np.float32)

    x_top = np.asarray(pts["x_top"], dtype=np.float32)
    y_top = np.asarray(pts["y_top"], dtype=np.float32)

    x_bottom = np.asarray(pts["x_bottom"], dtype=np.float32)
    y_bottom = np.asarray(pts["y_bottom"], dtype=np.float32)

    x_left = np.asarray(pts["x_left"], dtype=np.float32)
    y_left = np.asarray(pts["y_left"], dtype=np.float32)

    x_right = np.asarray(pts["x_right"], dtype=np.float32)
    y_right = np.asarray(pts["y_right"], dtype=np.float32)

    # Same ordering as in your original script
    x = np.concatenate((x_inner, x_left, x_right, x_bottom, x_top)).astype(np.float32)
    y = np.concatenate((y_inner, y_left, y_right, y_bottom, y_top)).astype(np.float32)

    inputs = [x, y]
    inner_inputs = np.vstack((x_inner, y_inner)).astype(np.float32)

    # Rectangle boundary for plotting
    xp = np.array([0.0, L, L, 0.0, 0.0], dtype=float)
    yp = np.array([-h / 2, -h / 2, h / 2, h / 2, -h / 2], dtype=float)

    # ------------------------------------------------------------------
    # Build PINN model
    # ------------------------------------------------------------------
    pinn = PINN(
        x_boun_left=x_left,
        y_boun_left=y_left,
        x_boun_right=x_right,
        y_boun_right=y_right,
        x_boun_top=x_top,
        y_boun_top=y_top,
        x_boun_bottom=x_bottom,
        y_boun_bottom=y_bottom,
        inner_inputs=inner_inputs,
        L=L,
        traction_y=traction_y,
        E=E,
        mu=mu,
    )

    pinn.summary()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print("Starting Adam training...")
    loss_adam = train_with_adam(
        model=pinn,
        inputs=inputs,
        epochs=adam_epochs,
        lr=adam_lr,
    )

    print("Adam optimizer stopped and L-BFGS started...")
    loss_lbfgs = train_with_lbfgs(
        model=pinn,
        inputs=inputs,
        max_iterations=lbfgs_max_iterations,
    )

    loss_total = np.concatenate((loss_adam, loss_lbfgs))



    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    # Preferred method name
    if hasattr(pinn, "predict_fields"):
        prediction = pinn.predict_fields(inputs)
    else:
        # Falls back to your older custom predict() implementation
        prediction = pinn.predict(inputs)

    displacement_x = prediction["u_x"]
    displacement_y = prediction["u_y"]
    stress_xx = prediction["sig_xx"]
    stress_yy = prediction["sig_yy"]
    stress_xy = prediction["sig_xy"]
    x_updated = prediction["x_new"]
    y_updated = prediction["y_new"]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    stress_error_dict = compute_stress_errors(
        x=x,
        y=y,
        stress_xx=stress_xx,
        stress_yy=stress_yy,
        stress_xy=stress_xy,
        L=L,
    )

    exact_disp_dict = compute_exact_displacement(
        x=x,
        y=y,
        L=L,
    )

    relative_error_dict = compute_relative_stress_errors(
        stress_xx=stress_xx,
        stress_yy=stress_yy,
        stress_xy=stress_xy,
        exact_sigxx=stress_error_dict["exact_sigxx"],
        exact_sigyy=stress_error_dict["exact_sigyy"],
        exact_sigxy=stress_error_dict["exact_sigxy"],
    )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    plot_loss(loss_total=loss_total, save_dir=save_dir)
    
    plot_pinn_displacement(
        x_updated=x_updated,
        y_updated=y_updated,
        displacement_x=displacement_x,
        displacement_y=displacement_y,
        xp=xp,
        yp=yp,
        save_dir=save_dir,
        filename="displacement_xy.png",
    )

    plot_pinn_stress(
        x_updated=x_updated,
        y_updated=y_updated,
        stress_xx=stress_xx,
        stress_yy=stress_yy,
        stress_xy=stress_xy,
        xp=xp,
        yp=yp,
        save_dir=save_dir,
        filename="stress_components.png",
    )

    plot_absolute_pinn_vs_exact_stress_error(
        x_updated=x_updated,
        y_updated=y_updated,
        error_sigxx=stress_error_dict["error_sigxx"],
        error_sigyy=stress_error_dict["error_sigyy"],
        error_sigxy=stress_error_dict["error_sigxy"],
        xp=xp,
        yp=yp,
        save_dir=save_dir,
        filename="abs_error_stress_components.png",
    )

    plot_analytical_displacement(
        x_updated=x_updated,
        y_updated=y_updated,
        exact_disp_x=exact_disp_dict["exact_disp_x"],
        exact_disp_y=exact_disp_dict["exact_disp_y"],
        xp=xp,
        yp=yp,
        save_dir=save_dir,
        filename="exact_displacement_components.png",
    )

    plot_relative_stress_error(
        x_updated=x_updated,
        y_updated=y_updated,
        rel_error_sigxx=relative_error_dict["rel_error_sigxx"],
        rel_error_sigyy=relative_error_dict["rel_error_sigyy"],
        rel_error_sigxy=relative_error_dict["rel_error_sigxy"],
        xp=xp,
        yp=yp,
        save_dir=save_dir,
        filename="rel_error_stress_components.png",
    )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    end_time = time.time()

    print(f"\nTraining and postprocessing completed.")
    print(f"Figures saved in: {save_dir}")
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()