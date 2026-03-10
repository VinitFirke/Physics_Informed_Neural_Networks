# Physics_Informed_Neural_Networks
This project demonstrates how PINNs can be used as a mesh-free alternative for solving elasticity problems while preserving physical consistency through the loss formulation.

## Problem Setup

This project studies the **simple bending of a 2D square domain** under **plane stress conditions**. The **right edge is fixed**, while a **prescribed shear traction** is applied on the **left edge**:

$$
\sigma_{xy} = t_y, \qquad t_y = 0.01 \,\text{N/m}^2
$$

The objective is to use a **Physics-Informed Neural Network (PINN)** to approximate the displacement and stress fields of the domain.

## Methodology

The spatial coordinates $(x, y)$ are generated randomly:
- inside the square domain,
- and along its boundary.

These coordinates are used as the **inputs** to the neural network.

The network predicts the following **output fields**:

$$
u_x,\; u_y,\; \sigma_{xx},\; \sigma_{yy},\; \sigma_{xy}
$$

To enforce the physical behavior of the problem, the model uses:
- **equilibrium equations**,
- **strain-displacement relations**,
- and **plane stress constitutive equations**.

Using the predicted displacement field, the strain components

$$
\varepsilon_{xx},\; \varepsilon_{yy},\; \varepsilon_{xy}
$$

are computed, and from these, the corresponding stress relations are evaluated.

## Loss Function

The PINN is trained by minimizing residuals derived from the governing equations and boundary conditions.

The residuals are obtained by comparing:
- the **stress components predicted by the neural network**, and
- the **stress components computed from the constitutive relations**.

Mean Squared Error (MSE) losses are then defined for:
- equilibrium equations,
- displacement constraints,
- additional boundary conditions,
- prescribed boundary conditions.

The total loss is written as:

$$
\mathrm{Loss}_{\mathrm{total}} =
\mathrm{Loss}_{\mathrm{balance}\,x,y}
+
\mathrm{Loss}_{\mathrm{displacement}\,x,y}
+
\mathrm{Loss}_{\mathrm{additional\ BC}}
+
\mathrm{Loss}_{\mathrm{prescribed\ BC}}
$$

## Test Cases

The following test case is considered:

**Case 1:**  
The **weight factors in the individual loss terms** are varied, and the resulting model performance is compared. The best configuration is selected based on the **lowest absolute and relative errors** in the predicted stress components.

## Goal of the Project

The purpose of this project is to investigate how effectively a PINN can solve a **2D elasticity bending problem** and how the choice of **loss weighting** influences the accuracy of the predicted displacement and stress fields.
