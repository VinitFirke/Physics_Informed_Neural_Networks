import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from lbfgs import function_factory

def lr_scheduler(epoch, lr):
    if epoch % 200 == 0 and epoch != 0:
        lr *= 0.9
    return lr

def train_with_adam(model, inputs, epochs=5000, lr=1e-2):
    model.compile(optimizer=tf.keras.optimizers.Nadam(lr), run_eagerly=False)
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    history = model.fit(inputs, epochs=epochs, verbose=2, callbacks=[lr_schedule])
    return history.history["loss"]

def train_with_lbfgs(model, inputs, max_iterations=20000):
    func = function_factory(lbfgs_model=model, train_x=inputs)
    init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)

    results = tfp.optimizer.bfgs_minimize(
        value_and_gradients_function=func,
        initial_position=init_params,
        tolerance=1e-16,
        x_tolerance=1e-16,
        f_relative_tolerance=0,
        max_iterations=max_iterations,
        parallel_iterations=10,
        max_line_search_iterations=20,
    )

    func.assign_new_model_parameters(results.position)
    return func.history

def train_model(model, inputs, epochs=5000, lr=1e-2, max_iterations=20000):
    loss_adam = train_with_adam(model, inputs, epochs=epochs, lr=lr)
    print("Adam optimizer stopped and L-BFGS started")
    loss_lbfgs = train_with_lbfgs(model, inputs, max_iterations=max_iterations)
    loss_total = np.concatenate((loss_adam, loss_lbfgs))
    return loss_total