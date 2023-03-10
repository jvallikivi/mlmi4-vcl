from typing import Any, Union
import numpy as np
import torch
from torch import nn
from alg.kcenter import KCenter
from alg.vae import VariationalAutoencoder, train_variational_autoencoder, ConvolutionalVariationalAutoencoder

class CoresetConfig:
    method: str
    post_process_method: Union[str, None]
    size: int
    device: str

    vae_params: Union[Any, None] = None

    def __init__(self, method='random', post_process_method=None, task_coreset_size=40, device: str = None, **config):
        ## <Input validation> ##
        if method not in ['random', 'k-center', 'vae-k-center']:
            raise ValueError(f"Invalid coreset builder method: {method}")
        if post_process_method is not None:
            assert post_process_method in [
                'shared-emb-mult', 'shared-emb-mult-per-task']
        
        invalid_config_keys = []
        allowed_config_keys = ['vae_params']
        for key in config.keys():
            if key not in allowed_config_keys:
                invalid_config_keys.append(key)
        
        if len(invalid_config_keys) > 0:
            raise ValueError(f"Invalid additional config keys: {invalid_config_keys}. Allowed: {allowed_config_keys}")
        ## </Input validation> ##

        self.method = method
        self.post_process_method = post_process_method
        self.size = task_coreset_size
        self.device = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")

        if method == 'vae-k-center':
            if 'vae_params' in config:
                self.vae_params = config['vae_params']
            else:
                self.vae_params = {}


class Coreset:
    def __init__(self, config: CoresetConfig = None):
        self.config = config if config is not None else CoresetConfig()
        self.x, self.y, self.task_mask, self.mult_mask, self.last_coreset_idx = None, None, None, None, None

    def update(self, data_x, data_y, task_idx):
        if self.config.method == 'random':
            coreset_idx = np.random.choice(data_x.shape[0], self.config.size, False)
        elif self.config.method == 'k-center':
            data_ = data_x.cpu().detach().numpy()
            if data_x.dim() == 4:
                data_ = data_.reshape(data_.shape[0], -1)
            coreset_idx = np.array(KCenter(self.config.size).fit_transform(data_))
        elif self.config.method == 'vae-k-center':
            if data_x.dim() == 4:
                if len(self.config.vae_params) > 0:
                    print("Warning: vae-params not implemented for convolutional variational autoencoder.")
                model = ConvolutionalVariationalAutoencoder(
                    in_channels=data_x.shape[1], 
                    image_height=data_x.shape[2], 
                    image_width=data_x.shape[3], 
                    emb_dim=32).to(self.config.device)
                train_variational_autoencoder(
                    model, data_x, n_epochs=100, batch_size=64, verbose=True)
            else:
                model = VariationalAutoencoder(**self.config.vae_params).to(self.config.device)
                train_variational_autoencoder(
                    model, data_x, n_epochs=100, batch_size=1024, verbose=False)
            coreset_idx = np.array(KCenter(self.config.size).fit_transform(
                model(data_x).cpu().detach().numpy()))
        else:
            raise ValueError(f"Invalid coreset builder method: {self.config.method}")

        self.last_coreset_idx = coreset_idx
        train_idx = np.delete(np.arange(data_x.shape[0]), coreset_idx)
        new_x = data_x[coreset_idx]
        new_y = data_y[coreset_idx]
        task_mask = torch.ones((new_x.shape[0]), dtype=int) * task_idx
        data_x = data_x[train_idx]
        data_y = data_y[train_idx]

        if self.x == None:
            self.x = new_x
            self.y = new_y
            self.task_mask = task_mask
            self.mult_mask = torch.arange(0, len(new_x))
        else:
            self.x = torch.cat([new_x, self.x])
            self.y = torch.cat([new_y, self.y])
            self.task_mask = torch.cat([task_mask, self.task_mask])
            self.mult_mask = torch.arange(0, len(self.x))

        # Returns the training data with coreset data removed
        return data_x, data_y

    def post_process(self, task_idx, embedder):
        if self.config.post_process_method is None:
            return

        if self.config.post_process_method == 'shared-emb-mult' and embedder is not None:
            self.mult_mask = torch.arange(0, len(self.x))  # reset everything
            with torch.no_grad():
                embeddings = embedder.forward_MLE(
                    self.x, [nn.ReLU() for i in range(len(embedder.layers) - 1)] + [nn.Identity()])
                coreset_important_idx = torch.tensor(
                    KCenter(int(len(self.x)/2)).fit_transform(embeddings.cpu().detach().numpy()))
                self.mult_mask = torch.cat(
                    (self.mult_mask, coreset_important_idx))

        if self.config.post_process_method == 'shared-emb-mult-per-task' and embedder is not None:
            self.mult_mask = torch.arange(0, len(self.x))


if __name__ == '__main__':
    # Just a test
    from datasets import MultiTaskDataset
    dataset = MultiTaskDataset('split CIFAR-10', 'cuda')
    trainx, trainy, _, _ = dataset.get_task_dataset(0)

    coreset = Coreset(CoresetConfig(method='vae-k-center'))
    x, y = coreset.update(trainx, trainy, 0)
    print(x.shape)
    print(coreset.x.shape)
