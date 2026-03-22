# Federated Learning (FedAvg) Implementation

**Improvement #14**: Privacy-preserving distributed training using Federated Averaging (FedAvg) protocol.

## Overview

This module implements the **Federated Averaging (FedAvg)** algorithm from [McMahan et al. (ICML 2017)](https://arxiv.org/abs/1602.05629): *Communication-Efficient Learning of Deep Networks from Decentralized Data*.

FedAvg enables collaborative training of deep learning models across multiple clients (e.g., wafer-producing plants) without centralizing sensitive manufacturing data:

1. **Server** initializes a global model
2. Each **round**: Server samples subset of clients
3. Each **client** trains on local data (E epochs) with local SGD
4. Clients upload model updates to server
5. **Server** aggregates updates via weighted averaging
6. Server broadcasts updated global model

### Key Features

- **Privacy-Preserving**: Raw data never leaves clients; only model updates are shared
- **Communication-Efficient**: Reduces rounds of communication needed vs. SGD
- **Non-IID Support**: Handles realistic non-identical and independently distributed data
- **Learning Rate Scheduling**: Constant or exponential decay schedules
- **Checkpoint Management**: Save/load global models
- **Multiple Architectures**: Compatible with WaferCNN, ResNet-18, EfficientNet-B0

## Architecture

```
src/federated/
├── __init__.py                 # Module marker
├── fed_avg.py                  # Core FedAvg algorithm (634 lines)
├── server.py                   # Server script and orchestration (383 lines)
└── client.py                   # Client script and manager (334 lines)

test_federated.py               # Comprehensive test suite (349 lines)
FEDERATED_LEARNING.md           # This documentation
```

## Core Classes

### `FedAvgConfig` (Configuration)

Dataclass for federated training hyperparameters:

```python
from src.federated.fed_avg import FedAvgConfig

config = FedAvgConfig(
    num_rounds=10,              # Federation rounds
    clients_per_round=5,        # Clients sampled each round
    local_epochs=2,             # Local SGD epochs per client
    learning_rate=0.01,         # Global learning rate
    learning_rate_schedule='exponential',  # 'constant' or 'exponential'
    lr_decay_rate=0.99,         # Decay factor per round
    device='cpu',               # 'cuda' or 'cpu'
    seed=42,                    # Reproducibility
    verbose=True,               # Enable logging
)
```

**Attributes**:
- `num_rounds`: Number of federation rounds
- `clients_per_round`: Number of clients sampled per round (< total clients)
- `local_epochs`: SGD epochs on each client per round
- `learning_rate`: Initial global learning rate
- `learning_rate_schedule`: 'constant' (fixed) or 'exponential' (decay)
- `lr_decay_rate`: Decay factor λ for exponential: lr_t = lr_0 * λ^t
- `device`: 'cuda' or 'cpu'
- `seed`: Random seed for reproducibility
- `verbose`: Enable logging to console

### `FedAveragingClient` (Local Training)

Simulates a federated client performing local SGD:

```python
from src.federated.fed_avg import FedAveragingClient
from src.models import WaferCNN

client = FedAveragingClient(
    client_id=0,
    train_loader=train_loader,           # DataLoader for local data
    model=WaferCNN(num_classes=9),       # Model to train
    criterion=nn.CrossEntropyLoss(),     # Loss function
    local_epochs=2,                      # SGD epochs per round
    learning_rate=0.01,                  # Local learning rate
    device='cpu',
)

# Perform local training
loss, acc = client.train_local(current_lr=0.01)

# Get model update for server
weights = client.get_weights()

# Sync with global model from server
client.set_weights(global_weights)

# Optional: evaluate on local validation set
val_acc = client.validate_local()
```

**Key Methods**:
- `train_local(current_lr)`: Train for local_epochs epochs; returns (loss, accuracy)
- `get_weights()`: Return model state_dict for server aggregation
- `set_weights(weights)`: Load global model from server
- `validate_local()`: Evaluate on optional local validation set

**Attributes**:
- `num_samples`: Number of training samples on this client (for weighted averaging)
- `local_loss_history`: Per-epoch training losses
- `local_acc_history`: Per-epoch training accuracies

### `FedAveragingServer` (Aggregation & Coordination)

Orchestrates global model updates via weighted averaging:

```python
from src.federated.fed_avg import FedAveragingServer

server = FedAveragingServer(
    model=global_model,          # Initial global model
    clients=clients,             # List of FedAveragingClient objects
    config=fed_config,           # FedAvgConfig instance
    test_loader=test_loader,     # Optional test set for evaluation
)

# Execute full training
results = server.train()
# results = {
#     'global_model': trained_model,
#     'round_loss': [r1_loss, r2_loss, ...],
#     'round_acc': [r1_acc, r2_acc, ...],
#     'test_acc': [r1_test_acc, r2_test_acc, ...] (if test_loader provided),
#     'total_time': elapsed_seconds,
#     'communication_rounds': num_rounds,
# }

# Or execute one round at a time
for round_num in range(config.num_rounds):
    loss, acc, test_acc = server.train_round(round_num)
    print(f"Round {round_num}: Loss={loss:.4f}, Acc={acc:.4f}")

# Get trained model
final_model = server.get_global_model()

# Save/load checkpoints
server.save_checkpoint('fed_model.pth')
server.load_checkpoint('fed_model.pth')
```

**FedAvg Aggregation Formula** (Weighted by client sample size):

```
w_{t+1} = Sum(n_k / n_total * w_k^t)

where:
  n_k = number of samples on client k
  n_total = total samples across all clients
  w_k^t = model weights trained on client k at round t
```

**Key Methods**:
- `train()`: Execute all federation rounds; returns results dict
- `train_round(round_num)`: Execute single round; returns (loss, acc, test_acc)
- `select_clients(num_clients)`: Sample random subset of clients
- `aggregate_weights(client_indices, client_weights)`: Weighted averaging
- `get_learning_rate(round_num)`: Compute scheduled learning rate
- `evaluate_global()`: Test on test_loader
- `get_global_model()`: Return trained model
- `save_checkpoint(path)`: Save global model and metrics
- `load_checkpoint(path)`: Load from checkpoint

**Attributes**:
- `round_loss_history`: Per-round average training losses
- `round_acc_history`: Per-round average training accuracies
- `test_acc_history`: Per-round test accuracies (if test_loader provided)

### Helper Function: `create_federated_setup()`

Convenience function to initialize federated learning:

```python
from src.federated.fed_avg import create_federated_setup
from src.models import WaferCNN

# Create federated setup
server, clients = create_federated_setup(
    model_class=WaferCNN,
    client_loaders=[loader1, loader2, loader3],  # One per client
    test_loader=test_loader,                     # Optional
    criterion=nn.CrossEntropyLoss(),             # Optional
    config=FedAvgConfig(num_rounds=10),          # Optional
    device='cpu',
)

# Train
results = server.train()
```

## Server Script

Run federated training via CLI:

```bash
# Basic: 10 rounds, 5 clients per round, default settings
python -m src.federated.server --model cnn --num-rounds 10

# Custom hyperparameters
python -m src.federated.server \
    --model resnet \
    --num-rounds 20 \
    --clients-per-round 10 \
    --local-epochs 2 \
    --num-clients 20 \
    --lr 0.001 \
    --batch-size 32 \
    --device cuda

# Non-IID data partitioning (lower alpha = more non-IID)
python -m src.federated.server \
    --model cnn \
    --num-clients 10 \
    --non-iid-alpha 0.5 \
    --log-file metrics.json
```

**Arguments**:
- `--model`: 'cnn' | 'resnet' | 'efficientnet'
- `--num-rounds`: Federation rounds (default: 10)
- `--clients-per-round`: Clients sampled per round (default: 5)
- `--local-epochs`: Local SGD epochs (default: 1)
- `--num-clients`: Total clients (default: 10)
- `--lr`: Learning rate (default: 0.01)
- `--batch-size`: Batch size per client (default: 32)
- `--non-iid-alpha`: Dirichlet alpha for non-IID partitioning (default: None = IID)
- `--device`: 'cuda' | 'cpu' (default: cpu)
- `--seed`: Random seed (default: 42)
- `--checkpoint-dir`: Save directory (default: 'checkpoints')
- `--log-file`: JSON file for metrics (optional)

### Data Partitioning

**IID Partitioning** (balanced):
```
Each client gets random equal-sized subset → Similar class distributions
```

**Non-IID Partitioning** (Dirichlet):
```
Use --non-iid-alpha with lower values for more non-IID:
  α = ∞ → perfectly IID (uniform class distribution)
  α = 0.5 → highly non-IID (clients specialize in subsets of classes)
  α = 0.1 → extremely non-IID (few classes per client)
```

## Client Script

Simulate federated client training:

```bash
# Simulate client 0 with CNN, 10 rounds
python -m src.federated.client --client-id 0 --model-type cnn --num-rounds 10

# Custom settings
python -m src.federated.client \
    --client-id 1 \
    --model-type resnet \
    --num-rounds 20 \
    --local-epochs 2 \
    --lr 0.001 \
    --output client_1_history.json
```

**Arguments**:
- `--client-id`: Client identifier (default: 0)
- `--model-type`: 'cnn' | 'resnet' | 'efficientnet' (default: cnn)
- `--num-rounds`: Federation rounds to simulate (default: 10)
- `--local-epochs`: Local SGD epochs per round (default: 2)
- `--lr`: Learning rate (default: 0.01)
- `--device`: 'cuda' | 'cpu' (default: cpu)
- `--output`: JSON file to save training history (optional)

## Usage Examples

### Example 1: Basic Federated Training

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset

from src.federated.fed_avg import (
    FedAvgConfig,
    FedAveragingServer,
    FedAveragingClient,
)
from src.models import WaferCNN

# Create dummy dataset
X = torch.randn(100, 3, 96, 96)
y = torch.randint(0, 9, (100,))
dataset = TensorDataset(X, y)

# Partition across 5 clients
client_loaders = []
for i in range(5):
    indices = list(range(i * 20, (i + 1) * 20))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=10, shuffle=True)
    client_loaders.append(loader)

# Create clients
criterion = nn.CrossEntropyLoss()
clients = []
for client_id, loader in enumerate(client_loaders):
    client = FedAveragingClient(
        client_id=client_id,
        train_loader=loader,
        model=WaferCNN(num_classes=9),
        criterion=criterion,
        local_epochs=2,
        learning_rate=0.01,
    )
    clients.append(client)

# Create server
config = FedAvgConfig(
    num_rounds=10,
    clients_per_round=3,
    local_epochs=2,
    learning_rate=0.01,
)
global_model = WaferCNN(num_classes=9)
server = FedAveragingServer(global_model, clients, config)

# Train
results = server.train()
print(f"Final accuracy: {results['round_acc'][-1]:.4f}")
```

### Example 2: Non-IID Training with Scheduling

```python
from src.federated.fed_avg import FedAvgConfig, create_federated_setup

# Create setup with non-IID data
server, clients = create_federated_setup(
    model_class=WaferCNN,
    client_loaders=client_loaders,
    config=FedAvgConfig(
        num_rounds=20,
        clients_per_round=5,
        local_epochs=2,
        learning_rate=0.1,
        learning_rate_schedule='exponential',  # Exponential decay
        lr_decay_rate=0.95,                    # 0.95^t decay
        verbose=True,
    ),
)

results = server.train()
print(f"Communication rounds: {results['communication_rounds']}")
print(f"Total time: {results['total_time']:.1f}s")
```

### Example 3: Checkpoint and Resume

```python
# Train and save
server.train()
server.save_checkpoint('fed_cnn_round_10.pth')

# Resume from checkpoint
server2 = FedAveragingServer(model, clients, config)
server2.load_checkpoint('fed_cnn_round_10.pth')
# Continue training
for i in range(10, 20):
    server2.train_round(i)
```

## Testing

Comprehensive test suite validates all components:

```bash
# Run all tests
python test_federated.py

# Expected output:
# Test 1: FedAvgConfig validation
#   PASS: Valid config created
#   PASS: Rejects num_rounds < 1
#   PASS: Rejects negative learning_rate
# Test 2: Client local training
#   PASS: Local training complete (loss=..., acc=...)
# ... (7 tests total)
# All tests PASSED
```

**Test Coverage**:
1. Configuration validation
2. Client local training
3. Server aggregation
4. End-to-end training loop
5. Learning rate scheduling
6. Non-IID data partitioning
7. Checkpoint save/load

## Architecture Decision Rationale

### Weighted Averaging
Models are aggregated using sample-size weighting:
```
w_avg = Sum(n_k / n_total * w_k)
```
This ensures clients with more data have proportionally more influence, preventing small clients from drowning out larger ones.

### Local SGD
Each client performs full SGD for `local_epochs` epochs on local data. This reduces communication rounds while maintaining convergence (McMahan et al. ICML 2017).

### Learning Rate Scheduling
- **Constant**: Fixed LR for all rounds → stable but may converge slowly
- **Exponential**: LR decays per round → faster early convergence, finer tuning late

### Non-IID Support
Dirichlet-based partitioning creates realistic scenarios where different clients have different label distributions (e.g., Plant A specializes in Center defects, Plant B in Donut).

### Privacy
Only model updates are shared (not raw data), enabling use cases where:
- Manufacturing plants cannot share raw wafer images
- Data is proprietary or regulated (GDPR, etc.)
- Competitive advantage requires data privacy

## Performance Characteristics

### Communication Cost
- **Total messages**: `num_rounds × clients_per_round × 2` (upload + download)
- **Message size**: ~500KB per model update (WaferCNN)
- **Savings vs. centralized SGD**: 10-100x fewer communication rounds (McMahan et al.)

### Computation Cost
- **Per client per round**: `local_epochs × |local_data| / batch_size` forward/backward passes
- **Server aggregation**: O(num_parameters) for weighted averaging
- **Wall-clock time**: Dominated by client computation (parallelizable)

### Convergence
From McMahan et al. (ICML 2017):
- FedAvg converges to stationary point for convex and non-convex objectives
- Convergence slower than centralized SGD (expected due to non-IID data)
- Mitigated by larger `local_epochs` and appropriate `learning_rate`

## Limitations and Future Work

### Current Limitations
1. **Synchronous aggregation**: Server waits for slowest client (straggler problem)
2. **No client dropout**: Assumes all sampled clients complete training
3. **Homogeneous models**: All clients train same architecture
4. **No differential privacy**: Model updates could leak information (requires DP-SGD layer)
5. **Single-machine server**: No server-side federation/decentralization

### Future Enhancements
1. **Asynchronous FedAvg**: Allow straggler clients to skip rounds
2. **Differential Privacy**: Add DP-SGD for formal privacy guarantees
3. **Model personalization**: Allow clients to fine-tune local heads
4. **Clustering**: Group similar clients; federate within clusters
5. **Byzantine-Robust Aggregation**: Handle malicious/corrupted updates
6. **Communication Compression**: Quantize/sparsify model updates

## References

1. **McMahan et al.** (ICML 2017): "Communication-Efficient Learning of Deep Networks from Decentralized Data"
   - Original FedAvg algorithm
   - Convergence analysis
   - Communication-efficiency improvements

2. **Kairouz et al.** (JMLR 2021): "Advances and Open Problems in Federated Learning"
   - Comprehensive survey
   - Privacy, fairness, robustness considerations

3. **Bonawitz et al.** (AISTATS 2019): "Towards Federated Learning at Scale: System Design"
   - TensorFlow Federated system design
   - Practical deployment lessons

## Code Quality & Standards

- **Type Hints**: All functions have parameter and return type annotations
- **Docstrings**: Comprehensive docstrings with algorithm details
- **Error Handling**: Validation in `__post_init__`, meaningful error messages
- **Reproducibility**: Fixed random seeds, stratified partitioning
- **Testing**: 7 comprehensive tests covering all core functionality
- **Logging**: INFO-level logging with structured messages

## Files Overview

| File | Lines | Purpose |
|------|-------|---------|
| `src/federated/__init__.py` | 6 | Module marker + docstring |
| `src/federated/fed_avg.py` | 634 | Core FedAvg classes: FedAvgConfig, FedAveragingClient, FedAveragingServer |
| `src/federated/server.py` | 383 | Server script, CLI entry point, data partitioning |
| `src/federated/client.py` | 334 | Client script, ClientManager, simulation utility |
| `test_federated.py` | 349 | 7 comprehensive tests |
| **Total** | **1706** | Complete implementation |

## Integration with Existing Project

The federated learning module integrates seamlessly:

```python
# Works with existing models
from src.models import WaferCNN, get_resnet18, get_efficientnet_b0
from src.federated.fed_avg import create_federated_setup

# Works with existing data
from src.data import load_dataset, preprocess_wafer_maps, WaferMapDataset

# Works with existing config
from src.training.config import TrainConfig

# Works with existing analysis
from src.analysis.evaluate import evaluate_model
```

## Conclusion

This federated learning implementation provides a production-ready FedAvg algorithm for privacy-preserving distributed training of wafer defect classification models. The implementation:

- Follows McMahan et al. (ICML 2017) precisely
- Supports multiple architectures and data partitioning strategies
- Includes comprehensive testing and documentation
- Integrates with existing project structure
- Enables realistic multi-plant collaboration scenarios

---

**Last Updated**: 2026-03-22
**Status**: Complete (Improvement #14/23)
