from prml import autodiff, nn


class Autoencoder(object):

    def __init__(self, *n_units):
        self.n_layers = len(n_units)
        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        for ndim_in, ndim_out in zip(n_units, n_units[1:]):
            self.encoder.append(nn.layers.Dense(ndim_in, ndim_out))
        for ndim_in, ndim_out in zip(n_units[::-1], n_units[-2::-1]):
            self.decoder.append(nn.layers.Dense(ndim_in, ndim_out))
        self.parameter = {}
        self.parameter.update({
            "encoder." + key: value for key, value in self.encoder.items()})
        self.parameter.update({
            "decoder." + key: value for key, value in self.decoder.items()})

    def transform(self, x):
        return self.encoder(x).value

    def fit(self, x, n_iter=100, learning_rate=1e-3):
        optimizer = nn.optimizer.Adam(self.parameter, learning_rate)
        for _ in range(n_iter):
            x_ = self.decoder(self.encoder(x))
            log_likelihood = autodiff.random.gaussian_logpdf(x_, x, 1).mean()
            optimizer.maximize(log_likelihood)
