# Federated Learning Guide

## Overview
This repository includes a robust implementation of Federated Learning for decentralized wafer defect detection.

## Key Features
1. **FedAvg Protocol**: Standard federated averaging algorithm.
2. **Byzantine Robustness**: Defense against poisoned client updates via `median`, `trimmed_mean`, or `krum` aggregation methods.
3. **Decentralized Execution**: Simulates $N$ independent clients performing localized SGD.

## Configuration
In `config.yaml` or when instantiating `FedAvgConfig`, set the `aggregation_method`:
```yaml
federated:
  num_rounds: 10
  clients_per_round: 5
  local_epochs: 2
  learning_rate: 0.01
  aggregation_method: "median"  # options: weighted_avg, median, trimmed_mean, krum
```

## Quick Start
```python
from src.federated.fed_avg import create_federated_setup, FedAvgConfig
from src.models import get_resnet18

# 1. Prepare configuration
config = FedAvgConfig(num_rounds=5, aggregation_method='trimmed_mean')

# 2. Setup server and clients
server, clients = create_federated_setup(
    model_class=get_resnet18,
    client_loaders=[train_loader_1, train_loader_2],
    test_loader=test_loader,
    config=config
)

# 3. Train
results = server.train()
print("Final Accuracy:", results['test_acc'][-1])
```