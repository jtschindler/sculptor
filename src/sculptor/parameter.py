import os
import pickle


class FitParameter:
    def __init__(self, name=None, value=None, vary=True, prior=None):
        self.name = name
        self.value = value
        self.vary = vary
        self.prior = prior

    def __str__(self):
        return f"{self.name}: {self.value}"

    def save(self, save_dir, save_prefix=None):

        # Check if the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if save_prefix is not None:
            save_name = f"{save_prefix}_{self.name}.pkl"
        else:
            save_name = f"{self.name}.pkl"

        # Save the object
        with open(os.path.join(save_dir, save_name), "wb") as f:
            pickle.dump(self, f)

    def load(self, name, load_dir, load_prefix=None):

        # Load the object
        if load_prefix is not None:
            load_name = f"{load_prefix}_{name}.pkl"
        else:
            load_name = f"{name}.pkl"

        with open(os.path.join(load_dir, load_name), "rb") as f:
            loaded_data = pickle.load(f)

            self.__dict__ = loaded_data.__dict__
