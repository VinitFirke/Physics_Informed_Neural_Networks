from libraries import *
def plot_loss(loss_total, 
    save_dir = "figures",
    filename = "PINN_stress_components.png"
    ):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure()
    plt.semilogy(loss_total)
    plt.xlabel("Total Iterations")
    plt.ylabel("Total Loss")
    plt.title("Adam Optimizer + L-BFGS Optimizer")
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()

#Plots PINN predicted displacement x and y
def plot_pinn_displacement(
        xp, yp,
        x_updated, y_updated,
        displacement_x, displacement_y,
        save_dir = "figures", filename = "PINN_disp_x_y.png"
        ): 
    os.makedirs(save_dir, exist_ok=True)
    x_updated = np.asarray(x_updated)
    y_updated = np.asarray(y_updated)
    displacement_x = np.asarray(displacement_x)
    displacement_y = np.asarray(displacement_y)

    fig, axes = plt.subplots(1, 2, figsize = (14,5))

    #Plotting displacement in x direction
    axes[0].plot(xp, yp, color="black", linewidth=2.5)
    scatter_ux = axes[0].scatter(
        x_updated,
        y_updated,
        c=displacement_x,
        cmap="RdBu_r",
    )
    cbar_ux = fig.colorbar(scatter_ux, ax=axes[0], format="%.2e")
    cbar_ux.set_label("Displacement in x")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Displacement in x (m)")

    #Plotting displacement in y direction
    axes[1].plot(xp, yp, color="black", linewidth=2.5)
    scatter_uy = axes[1].scatter(
        x_updated,
        y_updated,
        c=displacement_y,
        cmap="RdBu_r",
    )
    cbar_uy = fig.colorbar(scatter_uy, ax=axes[1], format="%.2e")
    cbar_uy.set_label("Displacement in y")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Displacement in y (m)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()

#Plots PINN predicted stresses (x, y, xy)
def plot_pinn_stress(
        x_updated, y_updated,
        stress_xx, stress_yy, stress_xy,
        xp, yp, 
        save_dir = "figures", filename = "PINN_stress_components.png"
):
    os.makedirs(save_dir, exist_ok=True)

    x_updated = np.asarray(x_updated)
    y_updated = np.asarray(y_updated)
    stress_xx = np.asarray(stress_xx)
    stress_yy = np.asarray(stress_yy)
    stress_xy = np.asarray(stress_xy)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Stress xx
    axes[0].plot(xp, yp, color="black", linewidth=2.5)
    scatter_sigxx = axes[0].scatter(
        x_updated,
        y_updated,
        c=stress_xx,
        cmap="RdBu_r",
    )
    cbar_sigxx = fig.colorbar(scatter_sigxx, ax=axes[0], format="%.2e")
    cbar_sigxx.set_label("Stress (xx)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Stress (xx) [N/m²]")

    # Stress yy
    axes[1].plot(xp, yp, color="black", linewidth=2.5)
    scatter_sigyy = axes[1].scatter(
        x_updated,
        y_updated,
        c=stress_yy,
        cmap="RdBu_r",
    )
    cbar_sigyy = fig.colorbar(scatter_sigyy, ax=axes[1], format="%.2e")
    cbar_sigyy.set_label("Stress (yy)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Stress (yy) [N/m²]")

    # Stress xy
    axes[2].plot(xp, yp, color="black", linewidth=2.5)
    scatter_sigxy = axes[2].scatter(
        x_updated,
        y_updated,
        c=stress_xy,
        cmap="RdBu_r",
    )
    cbar_sigxy = fig.colorbar(scatter_sigxy, ax=axes[2], format="%.2e")
    cbar_sigxy.set_label("Stress (xy)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("Stress (xy) [N/m²]")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()



#Plotting updated points after deformation
def plot_updated_points(
        xp, yp,
        x_updated, y_updated,
        save_dir = "figures", filename = "displaced_points.png"
        
):
    plt.figure(6)
    plt.plot(xp, yp, color="black", linewidth=2.5)
    plt.scatter(x_updated, y_updated)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Updated position of particles")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_absolute_pinn_vs_exact_stress_error(
            x_updated, y_updated,
            error_sigxx, error_sigyy, error_sigxy,
            xp, yp,
            save_dir = "figures",
            filename = "abs_error_stress_components_PINNvsEXACT.png"
        ):
        os.makedirs(save_dir, exist_ok=True)

        x_updated = np.asarray(x_updated)
        y_updated = np.asarray(y_updated)
        error_sigxx = np.asarray(error_sigxx)
        error_sigyy = np.asarray(error_sigyy)
        error_sigxy = np.asarray(error_sigxy)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        # Error sigma_xx
        axes[0].plot(xp, yp, color="black", linewidth=2.5)
        scatter_xx = axes[0].scatter(
            x_updated,
            y_updated,
            c=error_sigxx,
            cmap="RdBu_r",
        )
        cbar_xx = fig.colorbar(scatter_xx, ax=axes[0], format="%.2e")
        cbar_xx.set_label("Absolute error (Stress(xx))")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Absolute error (Stress(xx))")

        # Error sigma_yy
        axes[1].plot(xp, yp, color="black", linewidth=2.5)
        scatter_yy = axes[1].scatter(
            x_updated,
            y_updated,
            c=error_sigyy,
            cmap="RdBu_r",
        )
        cbar_yy = fig.colorbar(scatter_yy, ax=axes[1], format="%.2e")
        cbar_yy.set_label("Absolute error (Stress(yy))")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("Absolute error (Stress(yy))")

        # Error sigma_xy
        axes[2].plot(xp, yp, color="black", linewidth=2.5)
        scatter_xy = axes[2].scatter(
            x_updated,
            y_updated,
            c=error_sigxy,
            cmap="RdBu_r",
        )
        cbar_xy = fig.colorbar(scatter_xy, ax=axes[2], format="%.2e")
        cbar_xy.set_label("Absolute error (Stress(xy))")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title("Absolute error (Stress(xy))")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()

def plot_exact_stress(
            xp, yp,
            exact_sigxx, exact_sigyy, exact_sigxy,
            x_updated, y_updated,
            save_dir ="figures",
            filename = "analytical_stress_plot.png"
):
        os.makedirs(save_dir, exist_ok=True)

        x_updated = np.asarray(x_updated)
        y_updated = np.asarray(y_updated)
        exact_sigxx = np.asarray(exact_sigxx)
        exact_sigyy = np.asarray(exact_sigyy)
        exact_sigxy = np.asarray(exact_sigxy)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        axes[0].plot(xp, yp, color="black", linewidth=2.5)
        scatter_ea_sigxx = axes[0].scatter(x_updated, y_updated, c=exact_sigxx, cmap="RdBu_r")
        cbar_ea_sigxx = fig.colorbar(scatter_ea_sigxx, ax=axes[0], format="%.2e")
        cbar_ea_sigxx.set_label("Exact Solution (Stress(xx))")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Exact Solution (Stress(xx))")
        

        axes[1].plot(xp, yp, color="black", linewidth=2.5)
        scatter_ea_sigyy = axes[1].scatter(x_updated, y_updated, c=exact_sigyy, cmap="RdBu_r")
        cbar_ea_sigyy = fig.colorbar(scatter_ea_sigyy, ax=axes[1], format="%.2e")
        cbar_ea_sigyy.set_label("Exact Solution (Stress(yy))")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("Exact Solution (Stress(yy))")
        

        #plt.figure(12)
        axes[2].plot(xp, yp, color="black", linewidth=2.5)
        scatter_ea_sigxy = axes[2].scatter(x_updated, y_updated, c=exact_sigxy, cmap="RdBu_r")
        cbar_ea_sigxy = fig.colorbar(scatter_ea_sigxy, ax=axes[2], format="%.2e")
        cbar_ea_sigxy.set_label("Exact Solution (Stress(xy))")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        axes[2].set_title("Exact Solution (Stress(xy))")
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()



def plot_analytical_displacement(
          xp, yp,
          x_updated, y_updated,
          exact_disp_x, exact_disp_y,
          save_dir = "figures",
          filename = "analytical_displacement.png"
):
        os.makedirs(save_dir, exist_ok= True)
        x_updated = np.asarray(x_updated)
        y_updated = np.asarray(y_updated)
        exact_disp_x = np.asarray(exact_disp_x)
        exact_disp_y = np.asarray(exact_disp_y)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Exact displacement x
        axes[0].plot(xp, yp, color="black", linewidth=2.5)
        scatter_dx = axes[0].scatter(
            x_updated,
            y_updated,
            c=exact_disp_x,
            cmap="RdBu_r",
        )
        cbar_dx = fig.colorbar(scatter_dx, ax=axes[0], format="%.2e")
        cbar_dx.set_label("Exact Solution (Displacement(x))")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].set_title("Exact Solution (Displacement(x))")

        # Exact displacement y
        axes[1].plot(xp, yp, color="black", linewidth=2.5)
        scatter_dy = axes[1].scatter(
            x_updated,
            y_updated,
            c=exact_disp_y,
            cmap="RdBu_r",
        )
        cbar_dy = fig.colorbar(scatter_dy, ax=axes[1], format="%.2e")
        cbar_dy.set_label("Exact Solution (Displacement(y))")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        axes[1].set_title("Exact Solution (Displacement(y))")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
        plt.close()


def plot_relative_stress_error_side_by_side(
    x_updated,
    y_updated,
    rel_error_sigxx,
    rel_error_sigyy,
    rel_error_sigxy,
    xp,
    yp,
    save_dir="figures",
    filename="rel_error_stress_components.png",
):

    os.makedirs(save_dir, exist_ok=True)

    x_updated = np.asarray(x_updated)
    y_updated = np.asarray(y_updated)

    rel_error_sigxx = np.ma.masked_invalid(np.asarray(rel_error_sigxx))
    rel_error_sigyy = np.ma.masked_invalid(np.asarray(rel_error_sigyy))
    rel_error_sigxy = np.ma.masked_invalid(np.asarray(rel_error_sigxy))

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Relative error sigma_xx
    axes[0].plot(xp, yp, color="black", linewidth=2.5)
    scatter_xx = axes[0].scatter(
        x_updated,
        y_updated,
        c=rel_error_sigxx,
        cmap="RdBu_r",
    )
    cbar_xx = fig.colorbar(scatter_xx, ax=axes[0], format="%.2e")
    cbar_xx.set_label("Relative error (Stress(xx))")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Relative error (Stress(xx))")

    # Relative error sigma_yy
    axes[1].plot(xp, yp, color="black", linewidth=2.5)
    scatter_yy = axes[1].scatter(
        x_updated,
        y_updated,
        c=rel_error_sigyy,
        cmap="RdBu_r",
    )
    cbar_yy = fig.colorbar(scatter_yy, ax=axes[1], format="%.2e")
    cbar_yy.set_label("Relative error (Stress(yy))")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Relative error (Stress(yy))")

    # Relative error sigma_xy
    axes[2].plot(xp, yp, color="black", linewidth=2.5)
    scatter_xy = axes[2].scatter(
        x_updated,
        y_updated,
        c=rel_error_sigxy,
        cmap="RdBu_r",
    )
    cbar_xy = fig.colorbar(scatter_xy, ax=axes[2], format="%.2e")
    cbar_xy.set_label("Relative error (Stress(xy))")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_title("Relative error (Stress(xy))")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()