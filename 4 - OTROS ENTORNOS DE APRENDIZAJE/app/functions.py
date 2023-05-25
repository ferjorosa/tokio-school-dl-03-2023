import numpy as np
import tensorflow as tf
from tensorflow import keras


def load_model(file_name):
    model = keras.models.load_model(file_name)
    return model


model = load_model("../bcw_model")


def predict(
    clump_thickness,
    uniformity_of_cell_size,
    uniformity_of_cell_shape,
    marginal_adhesion,
    single_epithelial_cell_size,
    bare_nuclei,
    bland_chromatin,
    normal_nucleoli,
    mitoses,
):
    if model is not None:
        info = [
            clump_thickness,
            uniformity_of_cell_size,
            uniformity_of_cell_shape,
            marginal_adhesion,
            single_epithelial_cell_size,
            bare_nuclei,
            bland_chromatin,
            normal_nucleoli,
            mitoses,
        ]

        instance = tf.expand_dims(np.array(info), axis=0)

        prediction = model.predict(instance)

        results = dict()
        results["prediction"] = str(prediction[0][0])

        return results, 200
    else:
        return {"No se ha cargado ningun modelo"}, 500
