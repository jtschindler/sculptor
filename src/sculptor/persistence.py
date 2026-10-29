"""Pickle-based saving and loading shared by the fit classes.

Only load files you trust: unpickling can execute arbitrary code.
"""

import os
import pickle


def _pickle_path(name, directory, prefix=None):
    filename = f"{prefix}_{name}.pkl" if prefix is not None else f"{name}.pkl"
    return os.path.join(directory, filename)


class PickleMixin:
    """Adds `save`/`load` with one signature to `FitParameter`,
    `FitComponent` and `FitModel`."""

    def save(self, name, save_dir, prefix=None):
        """Save the object to `<save_dir>/[<prefix>_]<name>.pkl`.

        :param name: File name without extension
        :type name: str
        :param save_dir: Directory to save into (created if missing)
        :type save_dir: str or os.PathLike
        :param prefix: Optional prefix, joined to the name with "_"
        :type prefix: str
        """

        os.makedirs(save_dir, exist_ok=True)
        with open(_pickle_path(name, save_dir, prefix), "wb") as f:
            pickle.dump(self, f)

    def load(self, name, load_dir, prefix=None):
        """Replace this object's state with the one saved in
        `<load_dir>/[<prefix>_]<name>.pkl`.

        :param name: File name without extension
        :type name: str
        :param load_dir: Directory to load from
        :type load_dir: str or os.PathLike
        :param prefix: Optional prefix, joined to the name with "_"
        :type prefix: str
        :raises TypeError: if the file holds an object of a different class
        """

        with open(_pickle_path(name, load_dir, prefix), "rb") as f:
            loaded = pickle.load(f)

        if not isinstance(loaded, type(self)):
            raise TypeError(
                f"{_pickle_path(name, load_dir, prefix)} holds a "
                f"{type(loaded).__name__}, not a {type(self).__name__}."
            )

        # Copy the loaded object's attributes rather than sharing its dict:
        # with a shared dict, collecting the loaded object emptied the
        # objects it held (Python 3.13: parameters and spectrum of a
        # loaded FitModel gone after gc.collect()). Attributes that point
        # back to their owner (a FitComponent's param_mapping) are
        # re-pointed to this object.
        self.__dict__.clear()
        self.__dict__.update(loaded.__dict__)
        for value in self.__dict__.values():
            if getattr(value, "_owner", None) is loaded:
                value._owner = self
