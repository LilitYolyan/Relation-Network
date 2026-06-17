# Relation Network for Few-Shot Learning

A PyTorch implementation of **Relation Network (RN)** for few-shot image classification, based on [*Learning to Compare: Relation Network for Few-Shot Learning*](https://arxiv.org/abs/1711.06025) (Sung et al., CVPR 2018).

Instead of relying on a fixed distance metric (like Euclidean or cosine distance), this model learns the similarity metric itself: a small CNN ("relation module") looks at a pair of embeddings — one support example, one query example — and outputs a learned relation score between 0 and 1.

## How it works

The model has two parts, both defined in `model.py`:

- **`EmbeddingNet`** — a 4-block CNN feature extractor. Takes a batch of images `(N, 3, 28, 28)` and produces feature maps `(N, 64, 5, 5)`.
- **`RelationNet`** — the relation module. Takes a concatenated pair of support/query feature maps `(N, 128, 5, 5)` and outputs a relation score `(N, 1)` indicating how similar they are.

Training proceeds episodically:

1. Sample an N-way K-shot episode (a handful of support images per class, plus query images to classify).
2. Embed every image with `EmbeddingNet`.
3. Average the support embeddings per class into a single class prototype.
4. Pair every query embedding with every class prototype, concatenate them, and score the pair with `RelationNet`.
5. Train with MSE loss against one-hot relation labels (1 for the matching class, 0 otherwise).

This episodic sampling is handled by `task_sampler.py` and `loader.py`, and the full training loop lives in `runner.py`.

## Repository structure

| File | Description |
|---|---|
| `model.py` | `EmbeddingNet` and `RelationNet` module definitions |
| `loader.py` | Builds a `DataLoader` over an `ImageFolder`-style dataset using episodic sampling |
| `task_sampler.py` | Custom `Sampler` that draws N-way K-shot support/query batches per episode |
| `runner.py` | Training/validation loop, optimizer steps, and checkpointing |
| `train.py` | Entry point — wires up models, optimizers, schedulers, and config, then trains |
| `default.yaml` | Default configuration (episode size, paths, device, iteration count, etc.) |

## Requirements

- Python 3
- PyTorch
- torchvision
- PyYAML
- numpy
- tqdm

```bash
pip install torch torchvision pyyaml numpy tqdm
```

## Dataset format

The dataset loader expects a standard `torchvision.datasets.ImageFolder` layout, with one subfolder per class:

```
images_path/
├── class_1/
│   ├── img_001.png
│   └── img_002.png
├── class_2/
│   ├── img_001.png
│   └── img_002.png
└── ...
```

This works directly with datasets like **Omniglot** or **miniImageNet** once they're reorganized into this folder structure. Each class folder needs at least `shot + query_shot` images.

## Configuration

All hyperparameters are set in [`default.yaml`](./default.yaml):

```yaml
way: 5                  # classes per training episode
shot: 3                 # support images per class
query_way: 5            # classes per query set
query_shot: 3           # query images per class
images_size: 28         # image side length, in pixels
images_path: '/path/to/train/images'

val_way: 5
val_shot: 3
val_query_way: 5
val_query_shot: 3
val_images_path: '/path/to/val/images'

device: 'cuda'          # 'cuda' or 'cpu'
feature_dim: 64         # channel dimension of EmbeddingNet output
num_episodes: 10        # episodes sampled per training step
_max_iters: 10          # total training iterations
work_dir: '/path/to/checkpoints'
```

Update `images_path`, `val_images_path`, and `work_dir` to match your environment before training.

## Training

```bash
git clone https://github.com/LilitYolyan/Relation-Network.git
cd Relation-Network
python train.py
```

`train.py` reads `default.yaml`, builds the feature encoder and relation network, sets up SGD optimizers (lr `0.001`, momentum `0.9`) with a step learning-rate schedule, and trains via `Runner.run()`. Checkpoints are written periodically to `work_dir`.

## Reference

```bibtex
@inproceedings{sung2018learning,
  title={Learning to compare: Relation network for few-shot learning},
  author={Sung, Flood and Yang, Yongxin and Zhang, Li and Xiang, Tao and Torr, Philip HS and Hospedales, Timothy M},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1199--1208},
  year={2018}
}
```

## Notes

This is an educational/from-scratch implementation rather than an exact reproduction of the paper's reported numbers — architecture and training details (optimizer, schedule, episode count) can be tuned in `default.yaml` and `train.py` to match your dataset and compute budget.

## Contributing

Issues and pull requests are welcome.
