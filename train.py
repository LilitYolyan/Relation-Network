import torch
from model import RelationNet, EmbeddingNet
from runner import Runner
import yaml

feature_encoder = EmbeddingNet()
relation_network = RelationNet()

feature_encoder_optim = torch.optim.SGD(feature_encoder.parameters(),
                                        lr=0.001,
                                        momentum=0.9)
relation_network_optim = torch.optim.SGD(relation_network.parameters(),
                                         lr=0.001,
                                         momentum=0.9)

feature_encoder_scheduler = torch.optim.lr_scheduler.StepLR(
    feature_encoder_optim, step_size=30, gamma=0.1)
relation_network_scheduler = torch.optim.lr_scheduler.StepLR(
    relation_network_optim, step_size=30, gamma=0.1)

loss = torch.nn.MSELoss()

with open("./default.yaml", 'r') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

runner = Runner(feature_encoder, relation_network, feature_encoder_optim,
                relation_network_optim, feature_encoder_scheduler,
                relation_network_scheduler, loss, cfg)

runner.run()