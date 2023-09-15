# reference to https://github.com/Stability-AI/generative-models

from mindspore import ops


class UnitWeighting:
    def weighting(self, sigma):
        return ops.ones_like(sigma)

    def __call__(self, sigma):
        return self.weighting(sigma)


class EDMWeighting:
    def __init__(self, sigma_data=0.5):
        super(EDMWeighting, self).__init__()
        self.sigma_data = sigma_data

    def weighting(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

    def __call__(self, sigma):
        return self.weighting(sigma)


class VWeighting(EDMWeighting):
    def __init__(self):
        super().__init__(sigma_data=1.0)


class EpsWeighting:
    def weighting(self, sigma):
        return sigma ** -2.0

    def __call__(self, sigma):
        return self.weighting(sigma)
