from sculptor.persistence import PickleMixin


class FitParameter(PickleMixin):
    def __init__(self, name=None, value=None, vary=True, prior=None):
        self.name = name
        self.value = value
        self.vary = vary
        self.prior = prior

    def __str__(self):
        return f"{self.name}: {self.value}"
