import mindspore as ms
from mindspore import nn, ops
from mindspore.amp import all_finite

from gm.util import append_dims


class TrainNetwork(nn.Cell):
    def __init__(self, model, scaler):
        super(TrainNetwork, self).__init__()
        self.scaler = scaler

        self.model = model.model
        self.loss_fn = model.loss_fn
        self.denoiser = model.denoiser

    def construct(self, x, noised_input, sigmas, w, context, y):
        c_skip, c_out, c_in, c_noise = self.denoiser(sigmas, noised_input.ndim)
        model_output = self.model(
            ops.cast(noised_input * c_in, ms.float32),
            ops.cast(c_noise, ms.int32),
            context=context,
            y=y,
        )
        model_output = model_output * c_out + noised_input * c_skip
        loss = self.loss_fn(model_output, x, w)
        loss = loss.mean()
        return self.scaler.scale(loss)


class TrainOneStepSubCell(nn.Cell):
    def __init__(self, network, optimizer, reducer, scaler):
        super(TrainOneStepSubCell, self).__init__(auto_prefix=False)
        self.network = network
        self.optimizer = optimizer
        self.reducer = reducer
        self.scaler = scaler

        self.network.set_grad()
        self.grad_fn = ops.value_and_grad(network, grad_position=None, weights=optimizer.parameters)

    @ms.jit
    def construct(self, x, noised_input, sigmas, w, context, y):
        loss, grads = self.grad_fn(x, noised_input, sigmas, w, context, y)

        grads = self.reducer(grads)
        unscaled_grads = self.scaler.unscale(grads)
        grads_finite = all_finite(unscaled_grads)
        loss = ops.depend(loss, self.optimizer(unscaled_grads))
        return self.scaler.unscale(loss), unscaled_grads, grads_finite


class TrainOneStepCell(nn.Cell):
    def __init__(self, model, sub_cell):
        super(TrainOneStepCell, self).__init__()
        self.model = model
        self.encode_first_stage = model.encode_first_stage
        self.conditioner = model.conditioner
        self.sigma_sampler = model.sigma_sampler
        self.loss_fn = model.loss_fn
        self.denoiser = model.denoiser

        self.sub_cell = sub_cell

    def construct(self, x, tokens):
        # get latent
        x = self.encode_first_stage(x)
        cond = self.conditioner(tokens)
        cond = self.openai_input_warpper(cond)
        sigmas = self.sigma_sampler(x.shape[0])
        noise = ops.randn_like(x)
        noised_input = self.loss_fn.get_noise_input(x, noise, sigmas)
        w = append_dims(self.denoiser.w(sigmas), x.ndim)

        # get loss
        loss, _, _ = self.sub_cell(x, noised_input, sigmas, w, **cond)

        return loss

    def preprocess(self, batch):
        # get condition
        x = batch[self.model.input_key]
        tokens = self.model.conditioner.tokenize(batch)
        return x, tokens


class Trainer:
    def __init__(self, model, optimizer, reducer, scaler):
        train_network = TrainNetwork(model, scaler)
        train_one_step_sub_cell = TrainOneStepSubCell(train_network, optimizer, reducer, scaler)
        self.train_one_step_cell = TrainOneStepCell(model, train_one_step_sub_cell)

    def train_step(self, batch):
        x, tokens = self.train_one_step_cell.preprocess(batch)
        return self.train_one_step_cell(x, tokens)
