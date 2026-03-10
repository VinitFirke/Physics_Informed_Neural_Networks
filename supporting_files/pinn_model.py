################################################################

# Building the PINN in strong formulation based on 2D Linear elasticity Plane Stress

################################################################

from libraries import *
class PINN(tf.keras.Model):
    def __init__(
        self,
        x_boun_left=0.0,
        y_boun_left=0.0,
        x_boun_right=0.0,
        y_boun_right=0.0,
        x_boun_top=0.0,
        y_boun_top=0.0,
        x_boun_bottom=0.0,
        y_boun_bottom=0.0,
        inner_inputs=0.0,
        L=1.0,
        traction_y=0.01,
        E=1.0,
        mu=0.3,
    ):
        # Load Data in the class PINN

        super(PINN, self).__init__()

        self.E = E
        self.mu = mu

        self.x_boun_left = x_boun_left
        self.y_boun_left = y_boun_left
        self.x_boun_right = x_boun_right
        self.y_boun_right = y_boun_right
        self.x_boun_top = x_boun_top
        self.y_boun_top = y_boun_top
        self.x_boun_bottom = x_boun_bottom
        self.y_boun_bottom = y_boun_bottom

        self.inner_inputs = inner_inputs

        self.traction_y = traction_y

        self.L = L

        self.build(input_shape=2)

    def build(self, input_shape):
        # Building the ANN

        input_layer = tf.keras.layers.Input(shape=input_shape, name="input_layer")
        hidden_layers = 1  # Number of hidden layers

        previous_layer = input_layer

        for i in range(hidden_layers):
            layer_name = "hidden_layer_" + str(i)
            hidden_layer = tf.keras.layers.Dense(
                units=128,
                activation="tanh",
                kernel_initializer="glorot_normal",
                name=layer_name,
            )(previous_layer)
            previous_layer = hidden_layer

        final_output_layer = tf.keras.layers.Dense(
            units=5,
            name="output_layer",
        )(previous_layer)

        ux = final_output_layer[:, 0]
        uy = final_output_layer[:, 1]
        sigxx = final_output_layer[:, 2]
        sigyy = final_output_layer[:, 3]
        sigxy = final_output_layer[:, 4]

        self.ann = tf.keras.Model(
            inputs=[input_layer],
            outputs=[ux, uy, sigxx, sigyy, sigxy],
            name="ANN",
        )

        self.built = True

        return None

    def additionalLoss(self, xb, yb, xt, yt, xl, yl, xr, yr):
        # Additionally constraining the shear stresses on the top and bottom edges

        sigxy_bottom = self.ann(tf.stack((xb, yb), axis=1))[4]
        sigxy_top = self.ann(tf.stack((xt, yt), axis=1))[4]
        sigxy_left = self.ann(tf.stack((xl, yl), axis=1))[4]

        sigxx_left = self.ann(tf.stack((xl, yl), axis=1))[2]
        sigyy_top = self.ann(tf.stack((xt, yt), axis=1))[3]
        sigyy_bottom = self.ann(tf.stack((xb, yb), axis=1))[3]

        ux_right = self.ann(tf.stack((xr, yr), axis=1))[0]
        uy_right = self.ann(tf.stack((xr, yr), axis=1))[1]

        return {
            "sigxy_bottom": sigxy_bottom,
            "sigxy_top": sigxy_top,
            "sigxx_left": sigxx_left,
            "sigxy_left": sigxy_left,
            "sigyy_top": sigyy_top,
            "sigyy_bottom": sigyy_bottom,
            "ux_right": ux_right,
            "uy_right": uy_right,
        }

    def governing_equations(self, inputs):
        """
        Apply the BC's in a hard fashion to the outputs of the ANN before defining the governing equations
        Write the Governing equations here and return them with the help of a dictionary
        """

        x = inputs[0]
        y = inputs[1]

        with tf.GradientTape(persistent=True) as tape_x, tf.GradientTape(
            persistent=True
        ) as tape_y:
            tape_x.watch(x)
            tape_y.watch(y)

            disp_x = self.ann(tf.stack((x, y), axis=1))[0]
            disp_y = self.ann(tf.stack((x, y), axis=1))[1]
            stress_xx = self.ann(tf.stack((x, y), axis=1))[2]
            stress_yy = self.ann(tf.stack((x, y), axis=1))[3]
            stress_xy = self.ann(tf.stack((x, y), axis=1))[4]

            sigxx_x = tape_x.gradient(stress_xx, x)
            sigyy_y = tape_y.gradient(stress_yy, y)
            tau_x = tape_x.gradient(stress_xy, x)
            tau_y = tape_y.gradient(stress_xy, y)

            strain_x = tape_x.gradient(disp_x, x)
            strain_y = tape_y.gradient(disp_y, y)

            strain_x_y = tape_y.gradient(disp_x, y)
            strain_y_x = tape_x.gradient(disp_y, x)

            epsilon_xy = strain_x_y + strain_y_x

        del tape_x, tape_y

        return {
            "disp_x": disp_x,
            "disp_y": disp_y,
            "stress_x_x": sigxx_x,
            "stress_y_y": sigyy_y,
            "shear_stress_x": tau_x,
            "shear_stress_y": tau_y,
            "strain_x": strain_x,
            "strain_y": strain_y,
            "epsilon_xy": epsilon_xy,
            "stress_xx": stress_xx,
            "stress_yy": stress_yy,
            "stress_xy": stress_xy,
            "strain_x_y": strain_x_y,
            "strain_y_x": strain_y_x,
        }

    def residual(self, inputs):
        """
        Inside the residual the governing_equations() has to be called
        From the function governing_equations(), the returned dictionary has to be collected and the terms have to be added
        to form the total residual.
        """

        dict_governing_equations = self.governing_equations(inputs)
        sigmaxx_x = dict_governing_equations["stress_x_x"]
        sigmayy_y = dict_governing_equations["stress_y_y"]
        tau_x = dict_governing_equations["shear_stress_x"]
        tau_y = dict_governing_equations["shear_stress_y"]
        epsilon_x = dict_governing_equations["strain_x"]
        epsilon_y = dict_governing_equations["strain_y"]
        epsilon_xy = dict_governing_equations["epsilon_xy"]
        sigma_xx = dict_governing_equations["stress_xx"]
        sigma_yy = dict_governing_equations["stress_yy"]
        sigma_xy = dict_governing_equations["stress_xy"]

        # Residual: Governing equations

        balance_x = sigmaxx_x + tau_y
        balance_y = sigmayy_y + tau_x

        # Residual: Balance equations

        sigxx_p = (self.E / (1.0 - self.mu**2)) * (epsilon_x + self.mu * epsilon_y)
        sigyy_p = (self.E / (1.0 - self.mu**2)) * (self.mu * epsilon_x + epsilon_y)
        sigxy_p = (self.E / (2 * (1 + self.mu))) * (epsilon_xy)

        mat_xx = sigma_xx - sigxx_p
        mat_yy = sigma_yy - sigyy_p
        mat_xy = sigma_xy - sigxy_p

        # Strain compatability equations are missing - Check if they are needed

        return {
            "balance_x": balance_x,
            "balance_y": balance_y,
            "material_xx": mat_xx,
            "material_xy": mat_xy,
            "material_yy": mat_yy,
        }

    def call(self, inputs):
        residual = self.residual(inputs=inputs)

        bal_x = residual["balance_x"]
        bal_y = residual["balance_y"]
        mat_xx = residual["material_xx"]
        mat_yy = residual["material_yy"]
        mat_xy = residual["material_xy"]

        addShearLoss = self.additionalLoss(
            self.x_boun_bottom,
            self.y_boun_bottom,
            self.x_boun_top,
            self.y_boun_top,
            self.x_boun_left,
            self.y_boun_left,
            self.x_boun_right,
            self.y_boun_right,
        )

        Sigxy_top = addShearLoss["sigxy_top"]
        Sigxy_bottom = addShearLoss["sigxy_bottom"]
        Sigxy_left = addShearLoss["sigxy_left"]

        sigxx_left = addShearLoss["sigxx_left"]
        sigyy_top = addShearLoss["sigyy_top"]
        sigyy_bottom = addShearLoss["sigyy_bottom"]

        ux_right = addShearLoss["ux_right"]
        uy_right = addShearLoss["uy_right"]

        Residual_Sigxy_left = Sigxy_left - self.traction_y

        loss_w_e2 = 1e2
        loss_w_e3 = 1e3
        loss_w_e4 = 1e4

        loss_sigxy_bottom = tf.reduce_mean(tf.square(Sigxy_bottom))
        loss_sigxy_bottom = loss_w_e2 * loss_sigxy_bottom

        loss_sigxy_top = tf.reduce_mean(tf.square(Sigxy_top))
        loss_sigxy_top = loss_w_e2 * loss_sigxy_top

        loss_sigxy_left = tf.reduce_mean(tf.square(Residual_Sigxy_left))
        loss_sigxy_left = loss_w_e2 * loss_sigxy_left

        loss_bal_x = tf.reduce_mean(tf.square(bal_x))
        loss_bal_x = loss_w_e2 * loss_bal_x

        loss_bal_y = tf.reduce_mean(tf.square(bal_y))
        loss_bal_y = loss_w_e2 * loss_bal_y

        loss_mat_xx = tf.reduce_mean(tf.square(mat_xx))
        loss_mat_xx = loss_w_e2 * loss_mat_xx

        loss_mat_xy = tf.reduce_mean(tf.square(mat_xy))
        loss_mat_xy = loss_w_e2 * loss_mat_xy

        loss_mat_yy = tf.reduce_mean(tf.square(mat_yy))
        loss_mat_yy = loss_w_e2 * loss_mat_yy

        loss_sigxx_left = tf.reduce_mean(tf.square(sigxx_left))
        loss_sigxx_left = loss_w_e2 * loss_sigxx_left

        loss_sigyy_top = tf.reduce_mean(tf.square(sigyy_top))
        loss_sigyy_top = loss_w_e2 * loss_sigyy_top

        loss_sigyy_bottom = tf.reduce_mean(tf.square(sigyy_bottom))
        loss_sigyy_bottom = loss_w_e2 * loss_sigyy_bottom

        loss_ux_right = tf.reduce_mean(tf.square(ux_right))
        loss_ux_right = loss_w_e2 * loss_ux_right

        loss_uy_right = tf.reduce_mean(tf.square(uy_right))
        loss_uy_right = loss_w_e2 * loss_uy_right

        loss = (
            loss_bal_x
            + loss_bal_y
            + loss_mat_xx
            + loss_mat_xy
            + loss_mat_yy
            + loss_sigxy_bottom  # Shear Stress bottom: Natural NBC
            + loss_sigxy_top  # Shear Stress top: Natural NBC
            + loss_sigxy_left  # Shear Stress right: Prescribed NBC
            + loss_sigxx_left  # Sigxx: Prescribed NBC
            + loss_sigyy_top  # Sigyy: Natural NBC - top
            + loss_sigyy_bottom  # Sigyy: Natural NBC - bottom
            + loss_ux_right  # Displacement DBC ux
            + loss_uy_right  # Displacement DBC uy
        )

        loss_weight_factor = 1e2

        loss = loss * loss_weight_factor

        self.add_loss(loss)

        return loss

    def predict_fields(self, inputs):
        x = inputs[0]
        y = inputs[1]

        u_x, u_y, sig_xx, sig_yy, sig_xy = self.ann(tf.stack((x, y), axis=1))

        # Updated position of the particles

        x_new = x + u_x
        y_new = y + u_y

        return {
            "u_x": u_x,
            "u_y": u_y,
            "sig_xx": sig_xx,
            "sig_yy": sig_yy,
            "sig_xy": sig_xy,
            "x_new": x_new,
            "y_new": y_new,
        }
