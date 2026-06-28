import torch
import os
from loader import data_loader
import numpy as np
from tqdm import tqdm


class Runner:
    def __init__(
        self,
        feature_encoder,
        relation_network,
        feature_encoder_optim,
        relation_network_optim,
        feature_encoder_scheduler,
        relation_network_scheduler,
        loss_fn,
        config,
    ):
        self.feature_encoder = feature_encoder
        self.relation_network = relation_network
        self.feature_encoder_optim = feature_encoder_optim
        self.relation_network_optim = relation_network_optim
        self.feature_encoder_scheduler = feature_encoder_scheduler
        self.relation_network_scheduler = relation_network_scheduler
        self.loss_fn = loss_fn
        self.cfg = config
        self.total_rewards = {}
        self._iter = 0

    def run_episode(self, train_mode):
        """
            Run a single few-shot episode.
        
            Selects the training or validation config values (way, shot,
            query_way, query_shot, image size, and image path) depending on
            `train_mode`, builds an episodic data loader from `data_loader`,
            and feeds each sampled batch through `episode_processor` to
            compute the relation loss and prediction rewards for that episode.
        
            Args:
                train_mode (bool): If True, uses the training config keys
                    (`way`, `shot`, `query_way`, `query_shot`, `images_path`).
                    If False, uses the validation config keys (`val_way`,
                    `val_shot`, `val_query_way`, `val_query_shot`,
                    `val_images_path`).
        """
        if train_mode:
            way = self.cfg['way']
            shot = self.cfg['shot']
            query_way = self.cfg['query_way']
            query_shot = self.cfg['query_shot']
            image_size = self.cfg['images_size']
            image_path = self.cfg['images_path']

        else:
            way = self.cfg['val_way']
            shot = self.cfg['val_shot']
            query_way = self.cfg['val_query_way']
            query_shot = self.cfg['val_query_shot']
            image_size = self.cfg['images_size']
            image_path = self.cfg['val_images_path']

        loader = data_loader(way, shot, query_way, query_shot, image_size,
                             image_path)
        for data_batch in loader:
            self.episode_processor(data_batch, way, shot, query_way,
                                   query_shot)

    def train(self):
        """Run the training loop for one pass over the configured episodes.

            Moves the feature encoder and relation network to the configured
            device, sets them to train mode, and iterates over ``num_episodes``
            sampled episodes. For each episode it runs a forward pass via
            ``run_episode``, clears gradients on both networks, backpropagates the
            episode loss, and steps both optimizers. A checkpoint is written to the
            configured work directory every 1000 global iterations.
        
            Reads from ``self.cfg``:
                device (str | torch.device): Target device for the networks.
                num_episodes (int): Number of episodes to run in this call.
                work_dir (str): Directory where checkpoints are saved.
        
            Side effects:
                Sets ``self.mode`` to ``'train'``, updates ``self._inner_iter`` and
                ``self._iter``, mutates network and optimizer state in place, and may
                write checkpoint files to disk.
        
            Returns:
                None
        """
        self.feature_encoder.to(self.cfg['device']).train()
        self.relation_network.to(self.cfg['device']).train()
        self.mode = 'train'

        for i in range(self.cfg['num_episodes']):
            self._inner_iter = i
            self.run_episode(train_mode=True)
            self.feature_encoder.zero_grad()
            self.relation_network.zero_grad()
            self.loss.backward()
            self.feature_encoder_optim.step()
            self.relation_network_optim.step()
        self._iter += 1
        if self._iter % 1000 == 0:
            self.save_checkpoint(self.cfg['work_dir'])

    @torch.no_grad()
    def val(self):
        self.feature_encoder.eval()
        self.relation_network.eval()
        self.mode = 'val'
        for data_batch in range(self.cfg['num_episodes']):
            self._inner_iter = i
            self.run_episode(data_batch, train_mode=False)

    def run(self, **kwargs):
        for itr in tqdm(range(self.cfg['_max_iters'])):
            for i, flow in enumerate([('train', 1)]):
                mode, epochs = flow
                epoch_runner = getattr(self, mode)
                for _ in range(epochs):
                    if mode == 'train' and self._iter >= self.cfg['_max_iters']:
                        break
                    epoch_runner()

    def episode_processor(self, data_batch, way, shot, query_way, query_shot):
        inputs, labels = data_batch
        inputs = inputs.to(self.cfg['device'])
        # calculate features
        features = self.feature_encoder(inputs)
        classes = torch.unique(labels)
        support_idxs = torch.stack(
            list(map(lambda c: labels.eq(c).nonzero()[:shot],
                     classes))).view(-1)
        query_idxs = torch.stack(
            list(map(lambda c: labels.eq(c).nonzero()[shot:],
                     classes))).view(-1)
        support_features = features.to('cpu')[support_idxs]
        query_features = features.to('cpu')[query_idxs]
        support_labels = labels[support_idxs]
        query_labels = labels[query_idxs]
        f_map_width, f_map_height = tuple(support_features.shape[-2:])
        support_features = support_features.view(way, shot,
                                                 self.cfg['feature_dim'],
                                                 f_map_width, f_map_height)

        support_features = torch.mean(support_features, 1).squeeze(1)
        support_features_ext = support_features.unsqueeze(0).repeat(
            way * query_shot, 1, 1, 1, 1)
        query_features_ext = query_features.unsqueeze(0).repeat(
            way, 1, 1, 1, 1)
        query_features_ext = torch.transpose(query_features_ext, 0, 1)
        relation_pairs = torch.cat((support_features_ext, query_features_ext),
                                   2).view(-1, self.cfg['feature_dim'] * 2,
                                           f_map_width,
                                           f_map_height).to(self.cfg['device'])
        relations = self.relation_network(relation_pairs).view(
            -1, self.cfg['way']).to('cpu')
        query_labels_map = {
            int(j): i
            for i, j in enumerate(torch.unique(query_labels))
        }
        episode_labels = torch.tensor(
            [query_labels_map[int(i)] for i in query_labels])
        one_hot_labels = torch.zeros(query_shot * self.cfg['way'],
                                     way).scatter_(1,
                                                   episode_labels.unsqueeze(1),
                                                   1)
        self.loss = self.loss_fn(relations, one_hot_labels)
        _, predict_labels = torch.max(relations.data, 1)

        rewards = [
            1 if predict_labels[j] == query_labels[j] else 0
            for j in range(way)
        ]
        try:
            self.total_rewards[self.mode] += np.sum(rewards)
        except KeyError:
            self.total_rewards[self.mode] = np.sum(rewards)

    def save_checkpoint(
        self,
        out_dir,
    ):
        """
        Save the checkpoint.
        Args:
            out_dir (str): The directory that checkpoints are saved.
            
        """
        filename = f'weights_{self._iter}'
        filepath = osp.join(out_dir, filename)
        torch.save(
            {
                '_iter': self._iter,
                'feature_encode': self.feature_encoder.state_dict(),
                'relation_network': self.relation_network.state_dict(),
                'feature_encoder_optim':
                self.feature_encoder_optim.state_dict(),
                'relation_network_optim':
                self.relation_network_optim.state_dict(),
            }, filepath)
