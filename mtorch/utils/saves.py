import pickle
from mtorch.config import to_cpu, Device


def save_model(model, filepath="model_weights.pkl"):

    weights_to_save = [to_cpu(p.data) for p in mode.parameters()]

    with open(filepath, "wb") as f:
        pickle.dump(weights_to_save, f)

    print(f"Successfully saved weights to '{filepath}'")


def load_model(model, filepath=None):
    if filepath is None:
        raise ValueError("Must provide weights file name to load from")

    with open(filepath, "rb") as f:
        saved_weights = pickle.load(f)

    parameters = model.parameters()
    assert len(saved_weights) == len(
        parameters
    ), "Mismatch between saved weights and model's weights"

    for parameters, saved_array in zip(parameters, saved_weights):
        parameters.data = Device.xp.array(saved_array)

    print(f"Successfully loaded weights from '{filepath}'")
