from typing import List, Union
import torch
from torch import nn


class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_dim = in_features
        self.out_dim = out_features
        self.init_parameters()

    def init_parameters(self):
        self.weight_loc = nn.Parameter(torch.zeros(self.in_dim, self.out_dim))
        self.log_weight_scale = nn.Parameter(
            torch.zeros(self.in_dim, self.out_dim))

        self.bias_loc = nn.Parameter(torch.zeros(self.out_dim))
        self.log_bias_scale = nn.Parameter(torch.zeros(self.out_dim))

    def get_params(self):
        """
        return two tensors, obtaining by concatenating locs and scales together
        these parameters can be further used to calculate e.g, KL divergence (vectorizedly)
        """
        return (
            torch.cat([self.weight_loc.flatten(), self.bias_loc.flatten()]),
            torch.cat([self.log_weight_scale.flatten(),
                      self.log_bias_scale.flatten()])
        )

    def set_params(self, locs, scales):
        weight_loc, bias_loc = torch.split(
            locs.detach(), [self.in_dim * self.out_dim, self.out_dim])
        weight_loc = weight_loc.reshape(self.in_dim, self.out_dim)

        log_weight_scale, log_bias_scale = torch.split(
            scales.detach(), [self.in_dim * self.out_dim, self.out_dim])
        log_weight_scale = log_weight_scale.reshape(self.in_dim, self.out_dim)
        self.weight_loc.data = torch.clone(weight_loc)
        self.log_weight_scale.data = torch.clone(log_weight_scale)

        self.bias_loc.data = torch.clone(bias_loc)
        self.log_bias_scale.data = torch.clone(log_bias_scale)

    def forward(self, x, x_is_sample, activation_fn, n_particles):
        """
        forward with local reparameterization tricks
        """
        ys = []
        if x_is_sample:
            assert n_particles == len(x)
            for _x in x:
                gamma = _x @ self.weight_loc + self.bias_loc
                delta2 = (_x ** 2) @ (torch.exp(self.log_weight_scale)
                                      ** 2) + torch.exp(self.log_bias_scale) ** 2
                y = gamma + torch.randn_like(gamma) * torch.sqrt(delta2 + 1e-6)
                ys.append(activation_fn(y))
        else:
            for _ in range(n_particles):
                gamma = x @ self.weight_loc + self.bias_loc
                delta2 = (x ** 2) @ (torch.exp(self.log_weight_scale)
                                     ** 2) + torch.exp(self.log_bias_scale) ** 2
                y = gamma + torch.randn_like(gamma) * torch.sqrt(delta2 + 1e-6)
                ys.append(activation_fn(y))
        return ys

    def forward_MLE(self, x, activation_fn):
        y = x @ self.weight_loc + self.bias_loc
        return activation_fn(y)


class BayesNet(nn.Module):
    def __init__(self, in_features, hidden_features: Union[List[int], int], out_features: int):
        super(BayesNet, self).__init__()

        if type(hidden_features) is int:
            hidden_features = [hidden_features]
        else:
            assert type(hidden_features) is list

        if len(hidden_features) == 0:
            self.layers: List[BayesLinear] = nn.ModuleList(
                [BayesLinear(in_features, out_features)])
            self.layer_param_counts = [(in_features + 1) * out_features]
        else:
            self.layers: List[BayesLinear] = nn.ModuleList(
                [BayesLinear(in_features, hidden_features[0])]
                + [BayesLinear(hidden_features[i], hidden_features[i+1])
                   for i in range(len(hidden_features) - 1)]
                + [BayesLinear(hidden_features[-1], out_features)]
            )
            self.layer_param_counts = [
                (a + 1) * b for a, b in zip([in_features] + hidden_features, hidden_features + [out_features])]

    def forward(self, x: torch.Tensor, x_is_sample: bool, activation_fns: Union[List[nn.Module], nn.Module], n_particles):
        activation_fns = activation_fns if type(activation_fns) is list else [
            activation_fns for _ in range(len(self.layers))]

        x = self.layers[0](x, x_is_sample, activation_fns[0], n_particles)
        for i, layer in enumerate(self.layers[1:]):
            x = layer(x, True, activation_fns[i], n_particles)
        return x

    def forward_MLE(self, x: torch.Tensor, activation_fns: Union[List[nn.Module], nn.Module]):
        activation_fns = activation_fns if type(activation_fns) is list else [
            activation_fns for _ in range(len(self.layers))]
        for i, layer in enumerate(self.layers):
            x = layer.forward_MLE(x, activation_fns[i])
        return x

    def get_params(self):
        locs = []
        scales = []
        for layer in self.layers:
            layer_locs, layer_scales = layer.get_params()
            locs.append(layer_locs)
            scales.append(layer_scales)
        return torch.cat(locs), torch.cat(scales)

    def set_params(self, locs, scales):
        next_i = 0
        for count, layer in zip(self.layer_param_counts, self.layers):
            layer.set_params(locs[next_i:next_i+count],
                             scales[next_i:next_i+count])
            next_i += count

    def reinit_each_layer(self):
        device = self.layers[0].weight_loc.data.get_device()
        for layer in self.layers:
            layer.init_parameters()
        if device != -1:
            self.to(device)

    def reinit_each_layer_with(self, init_fn):
        """
            Initialises each layer by a given function. Caller must ensure that init_fn modifies in place.
        """
        device = self.layers[0].weight_loc.data.get_device()
        for layer in self.layers:
            init_fn(layer)
        if device != -1:
            self.to(device)


# just some tests here to ensure set_params/get_params is sound
def _test_bayes_linear_get_set():
    linear_1, linear_2 = BayesLinear(9, 15), BayesLinear(9, 15)

    linear_1.weight_loc.data = torch.randn_like(linear_1.weight_loc.data)
    linear_1.bias_loc.data = torch.randn_like(linear_1.bias_loc.data)
    linear_1.log_weight_scale.data = torch.randn_like(
        linear_1.log_weight_scale.data)
    linear_1.log_bias_scale.data = torch.randn_like(
        linear_1.log_bias_scale.data)

    locs_1, scales_1 = linear_1.get_params()
    locs_1, scales_1 = locs_1.clone(), scales_1.clone()

    linear_2.set_params(locs_1, scales_1)
    locs_2, scales_2 = linear_2.get_params()

    assert torch.allclose(locs_1, locs_2)
    assert torch.allclose(scales_1, scales_2)


def _test_deep_bayes_get_set_single_layer():
    net_1, net_2 = BayesNet(9, [], 15), BayesNet(9, [], 15)

    def init_layer(linear):
        linear.weight_loc.data = torch.randn_like(linear.weight_loc.data)
        linear.bias_loc.data = torch.randn_like(linear.bias_loc.data)
        linear.log_weight_scale.data = torch.randn_like(
            linear.log_weight_scale.data)
        linear.log_bias_scale.data = torch.randn_like(
            linear.log_bias_scale.data)
    net_1.init_each_layer_with(init_layer)

    locs_1, scales_1 = net_1.get_params()
    locs_1, scales_1 = locs_1.clone(), scales_1.clone()

    net_2.set_params(locs_1, scales_1)
    locs_2, scales_2 = net_2.get_params()

    assert torch.allclose(locs_1, locs_2)
    assert torch.allclose(scales_1, scales_2)


def _test_deep_bayes_get_set():
    net_1, net_2 = BayesNet(
        9, [15, 45, 23], 2), BayesNet(9, [15, 45, 23], 2)

    def init_layer(linear):
        linear.weight_loc.data = torch.randn_like(linear.weight_loc.data)
        linear.bias_loc.data = torch.randn_like(linear.bias_loc.data)
        linear.log_weight_scale.data = torch.randn_like(
            linear.log_weight_scale.data)
        linear.log_bias_scale.data = torch.randn_like(
            linear.log_bias_scale.data)
    net_1.init_each_layer_with(init_layer)

    locs_1, scales_1 = net_1.get_params()
    locs_1, scales_1 = locs_1.clone(), scales_1.clone()

    net_2.set_params(locs_1, scales_1)
    locs_2, scales_2 = net_2.get_params()

    assert torch.allclose(locs_1, locs_2)
    assert torch.allclose(scales_1, scales_2)


if __name__ == '__main__':
    print("Running tests.")
    _test_bayes_linear_get_set()
    _test_deep_bayes_get_set_single_layer()
    _test_deep_bayes_get_set()
    print("All tests successful.")
