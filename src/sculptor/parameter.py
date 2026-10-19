from sculptor.persistence import PickleMixin


class FitParameter(PickleMixin):
    """A single fit parameter.

    :param name: Parameter name (the key in the model's parameter dict)
    :type name: str
    :param value: Value; the starting value if the parameter varies
    :type value: float
    :param vary: Sample this parameter (True) or keep it fixed at `value`
    :type vary: bool
    :param prior: Prior for a varying parameter; also used to draw the
        initial walker positions
    :type prior: sculptor.prior.Prior
    """

    def __init__(self, name=None, value=None, vary=True, prior=None):
        self.name = name
        self.value = value
        self.vary = vary
        self.prior = prior

    def __str__(self):
        return f"{self.name}: {self.value}"
