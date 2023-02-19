from copy import deepcopy
from typing import List, Union
from alg.bayes import BayesLinear, BayesNet
import torch
from torch import nn
import numpy as np
from torch.distributions import kl_divergence, Normal
from collections import OrderedDict
from enum import Enum, auto


class Initialization(Enum):
    DEFAULT = auto()
    RANDOM = auto()
    SMALL_VARIANCE = auto()


def init_layer_small_variance(layer: BayesLinear):
    layer.weight_loc.data = torch.zeros_like(layer.weight_loc.data)
    layer.bias_loc.data = torch.zeros_like(layer.bias_loc.data)
    # initialize to very small value for the first model
    layer.log_weight_scale.data = torch.zeros_like(
        layer.log_weight_scale.data) - 3
    layer.log_bias_scale.data = torch.zeros_like(layer.log_bias_scale.data) - 3


def init_layer_by_random(layer: BayesLinear):
    layer.weight_loc.data = nn.init.normal_(
        torch.zeros_like(layer.weight_loc.data), std=0.1)
    layer.bias_loc.data = nn.init.normal_(
        torch.zeros_like(layer.bias_loc.data), std=0.1)
    layer.log_weight_scale.data = torch.zeros_like(
        layer.log_weight_scale.data) - 3
    layer.log_bias_scale.data = torch.zeros_like(
        layer.log_bias_scale.data) - 3


class MultiHeadVCL(nn.Module):
    def __init__(self, in_dim: int, shared_hidden_dims: List[int], head_hidden_dims: List[int], head_out_dim: int, num_heads=1, initialization: Initialization = Initialization.DEFAULT):
        super(MultiHeadVCL, self).__init__()

        self.in_dim = in_dim
        self.shared_hidden_dims = deepcopy(shared_hidden_dims)
        self.head_hidden_dims = deepcopy(head_hidden_dims)
        self.head_out_dim = head_out_dim

        self.shared = BayesNet(
            self.in_dim, self.shared_hidden_dims[:-1], self.shared_hidden_dims[-1])

        self.heads: List[BayesNet] = nn.ModuleList(
            [BayesNet(self.shared_hidden_dims[-1], self.head_hidden_dims, self.head_out_dim) for _ in range(num_heads)])

        # define a layer dict
        self.layer_dict = OrderedDict()
        self.layer_dict["shared"] = self.shared
        self.layer_dict["heads"] = self.heads

        with torch.no_grad():
            for key in self.layer_dict:
                if key == 'heads':
                    for head in self.layer_dict[key]:
                        if initialization is not Initialization.DEFAULT:
                            assert initialization is Initialization.RANDOM or initialization is Initialization.SMALL_VARIANCE
                            head.reinit_each_layer_with(
                                init_layer_by_random if initialization is Initialization.RANDOM else init_layer_small_variance)
                else:
                    if initialization is not Initialization.DEFAULT:
                        assert initialization is Initialization.RANDOM or initialization is Initialization.SMALL_VARIANCE
                        self.layer_dict[key].reinit_each_layer_with(
                            init_layer_by_random if initialization is Initialization.RANDOM else init_layer_small_variance)

    @staticmethod
    def new_from_prior(prior_model: 'MultiHeadVCL') -> 'MultiHeadVCL':
        device = prior_model.shared.layers[0].weight_loc.data.get_device()
        model = MultiHeadVCL(in_dim=prior_model.in_dim, shared_hidden_dims=prior_model.shared_hidden_dims, head_hidden_dims=prior_model.head_hidden_dims,
                             head_out_dim=prior_model.head_out_dim, num_heads=len(prior_model.heads)).to(device)
        with torch.no_grad():
            for key in model.layer_dict:
                if key == 'heads':
                    for head_i, head in enumerate(model.layer_dict[key]):
                        head.set_params(
                            *prior_model.layer_dict[key][head_i].get_params())
                else:
                    model.layer_dict[key].set_params(
                        *prior_model.layer_dict[key].get_params())

        model.set_prior(prior_model)
        return model

    def set_prior(self, prior_model: 'MultiHeadVCL'):
        with torch.no_grad():
            prior_locs, prior_logscales = prior_model.get_params()
            self.prior_model_locs = prior_locs.detach().clone()
            self.prior_model_log_scales = prior_logscales.detach().clone()

    def get_prior(self) -> 'MultiHeadVCL':
        if self.prior_model_locs is None:
            raise Exception("Prior not set!")
        with torch.no_grad():
            prior = MultiHeadVCL(
                random_initialize=False, num_heads=len(self.heads))
            prior.set_params(self.prior_model_locs,
                             self.prior_model_log_scales)
            return prior

    def init_shared_weights(self, initialization: Initialization):
        with torch.no_grad():
            if initialization is Initialization.DEFAULT:
                self.shared.reinit_each_layer()
            elif initialization is Initialization.RANDOM:
                self.shared.reinit_each_layer_with(init_layer_by_random)
            elif initialization is Initialization.SMALL_VARIANCE:
                self.shared.reinit_each_layer_with(init_layer_small_variance)
            else:
                raise ValueError(
                    "Unexpected initialization config: ", initialization)

    def add_head(self, initialization: Initialization):
        device = self.shared.layers[0].weight_loc.data.get_device()
        self.heads: List[BayesNet] = nn.ModuleList([head for head in self.heads] + [BayesNet(
            self.shared_hidden_dims[-1], self.head_hidden_dims, self.head_out_dim).to(device)])
        self.layer_dict["heads"] = self.heads

        # also add new head prior
        if self.prior_model_locs is not None:
            with torch.no_grad():
                locs, log_scales = self.heads[-1].get_params()
                self.prior_model_locs = torch.cat(
                    (self.prior_model_locs, locs.detach().clone()))
                self.prior_model_log_scales = torch.cat(
                    (self.prior_model_log_scales, log_scales.detach().clone()))

        with torch.no_grad():
            if initialization is Initialization.DEFAULT:
                self.heads[-1].reinit_each_layer()
            elif initialization is Initialization.RANDOM:
                self.heads[-1].reinit_each_layer_with(init_layer_by_random)
            elif initialization is Initialization.SMALL_VARIANCE:
                self.heads[-1].reinit_each_layer_with(
                    init_layer_small_variance)
            else:
                raise ValueError(
                    "Unexpected initialization config: ", initialization)

    def predict(self, x, task_i_mask: Union[int, np.int64, torch.Tensor], n_particles):
        device = x.get_device()
        hiddens = self.shared.forward(
            x, x_is_sample=False, activation_fns=nn.ReLU(), n_particles=n_particles)

        if type(task_i_mask) is int or type(task_i_mask) is np.int64:
            logits = self.heads[task_i_mask].forward(
                hiddens, x_is_sample=True, activation_fns=nn.Identity(), n_particles=n_particles)
            return logits
        else:
            assert type(task_i_mask) is torch.Tensor, type(task_i_mask)
            task_datapoints = [(task_i, task_i_mask == task_i)
                               for task_i in task_i_mask.unique().numpy()]
            logits_samples = [torch.zeros((x.shape[0], self.head_out_dim)).to(
                device) for i in range(n_particles)]
            for task_i, mask in task_datapoints:
                if task_i >= len(self.heads):
                    print(task_i, mask)
                    raise Exception("Invalid task mask")
                task_logits_samples = self.heads[task_i].forward(
                    [hidden[mask] for hidden in hiddens], x_is_sample=True, activation_fns=nn.Identity(), n_particles=n_particles)
                for logits, task_logits in zip(logits_samples, task_logits_samples):
                    logits[mask] = task_logits
            return logits_samples

    def predict_MLE(self, x, task_i_mask: Union[int, np.int64, torch.Tensor]):
        device = x.get_device()
        hiddens = self.shared.forward_MLE(x, nn.ReLU())
        if type(task_i_mask) is int or type(task_i_mask) is np.int64:
            logits = self.heads[task_i_mask].forward_MLE(
                hiddens, nn.Identity())
        else:
            assert type(task_i_mask) is torch.Tensor, type(task_i_mask)
            logits = torch.zeros((x.shape[0], self.head_out_dim)).to(device)
            task_datapoints = [(task_i, task_i_mask == task_i)
                               for task_i in task_i_mask.unique().numpy()]
            for task_i, mask in task_datapoints:
                logits[mask] = self.heads[task_i].forward_MLE(
                    [hidden[mask] for hidden in hiddens], nn.Identity())
        return logits

    def get_params(self):
        locs = []
        logscales = []
        for key in self.layer_dict:
            if key == 'heads':
                for head in self.layer_dict[key]:
                    loc, scale = head.get_params()
                    locs.append(loc)
                    logscales.append(scale)
            else:
                loc, scale = self.layer_dict[key].get_params()
                locs.append(loc)
                logscales.append(scale)
        locs = torch.cat(locs)
        logscales = torch.cat(logscales)
        return locs, logscales

    def set_params(self, locs, scales):
        locs, scales = locs.detach().clone(), scales.detach().clone()
        next_i = 0
        for key in self.layer_dict:
            if key == 'heads':
                for head in self.layer_dict[key]:
                    shape = head.weight_loc.shape
                    size = (shape[0] + 1) * shape[1]
                    head.set_params(locs[next_i:next_i+size],
                                    scales[next_i:next_i+size])
                    next_i += size
            else:
                shape = self.layer_dict[key].weight_loc.shape
                size = (shape[0] + 1) * shape[1]
                self.layer_dict[key].set_params(
                    locs[next_i:next_i+size], scales[next_i:next_i+size])
                next_i += size
        assert next_i == locs.shape[0] and next_i == scales.shape[
            0], f"{size}, {locs.shape}, {scales.shape}"

    def calculate_ELBO(self, x, y, task_i_mask: Union[torch.Tensor, int], n_particles, dataset_size):
        locs, logscales = self.get_params()
        KL = kl_divergence(Normal(loc=locs, scale=torch.exp(logscales)),
                           Normal(loc=self.prior_model_locs, scale=torch.exp(
                               self.prior_model_log_scales))
                           )

        nELBO = 0
        for _ in range(n_particles):
            logit = self.predict(x, task_i_mask, 1)[0]
            neg_log_p = nn.CrossEntropyLoss(reduction='sum')(logit, y)
            nELBO = neg_log_p + nELBO
        nELBO = nELBO / n_particles / x.shape[0] + KL.sum() / dataset_size
        return nELBO

    def MLE_loss(self, x, y, task_i_mask: Union[torch.Tensor, int]):
        logit = self.predict_MLE(x, task_i_mask)
        loss = nn.CrossEntropyLoss(reduction='mean')(logit, y)
        return loss


class MultiHeadVCLSplitMNIST(MultiHeadVCL):
    def __init__(self, num_heads=1, initialization: Initialization = Initialization.DEFAULT):
        # "We use fully connected multi-head networks with two hidden layers comprising 256 hidden units with ReLU activations."
        super(MultiHeadVCLSplitMNIST, self).__init__(
            in_dim=28*28, shared_hidden_dims=[256, 256], head_hidden_dims=[], head_out_dim=2, num_heads=num_heads, initialization=initialization)


class MultiHeadVCLSplitNotMNIST(MultiHeadVCL):
    def __init__(self, num_heads=1, initialization: Initialization = Initialization.DEFAULT):
        # "The settings for this experiment are the same as those in the Split MNIST experiment above, except that we use deeper networks with 4 hidden layers, each of which contains 150 hidden units."
        super(MultiHeadVCLSplitNotMNIST, self).__init__(
            in_dim=28*28, shared_hidden_dims=[150, 150, 150, 150], head_hidden_dims=[], head_out_dim=2, num_heads=num_heads, initialization=initialization)
