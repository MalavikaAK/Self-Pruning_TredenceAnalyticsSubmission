import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)


# Part 1: PrunableLinear Layer
# Each weight has a learnable gate_score of the same shape as the weight tensor.
# gate = sigmoid(gate_score) which is always in range (0, 1).
# pruned_weight = weight * gate (element-wise multiplication).
# When gate -> 0, the weight contributes nothing to the output (effectively pruned).
# Both weight and gate_scores are nn.Parameter so autograd tracks gradients for both.
# The bimodal regularizer term gate*(1-gate) pushes gates toward either 0 or 1.

class PrunableLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        # Standard learnable weight and bias parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # gate_scores: same shape as weight, registered as a learnable parameter.
        # Initialized to 2.0 so sigmoid(2.0) = 0.88, gates start open.
        # The sparsity loss will push unnecessary gates toward 0 during training.
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), 2.0))

        # Kaiming uniform initialization is best practice for ReLU-based networks
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x):
        # Step 1: squeeze gate_scores into (0, 1) via sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: mask each weight by its gate, if gate=0 weight is removed
        pruned_weights = self.weight * gates

        # Step 3: standard linear operation using pruned weights
        # autograd automatically computes gradients for both weight and gate_scores
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        # Returns detached gate values for evaluation and plotting (no grad needed)
        return torch.sigmoid(self.gate_scores).detach()


# Part 2a: Network Architecture using PrunableLinear
# All four linear layers are PrunableLinear, no standard nn.Linear used anywhere.
# Architecture: flatten(3072) -> 1024 -> 512 -> 256 -> 10 classes

class SelfPruningNet(nn.Module):

    def __init__(self):
        super().__init__()

        # Four prunable layers replacing all standard linear layers
        self.fc1 = PrunableLinear(3072, 1024)
        self.fc2 = PrunableLinear(1024, 512)
        self.fc3 = PrunableLinear(512,  256)
        self.fc4 = PrunableLinear(256,  10)

        # BatchNorm improves training stability after each hidden layer
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

        # Dropout(0.3) reduces overfitting during training
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)           # Flatten: (B,3,32,32) -> (B,3072)
        x = F.relu(self.bn1(self.fc1(x)))   # Hidden layer 1: 3072 -> 1024
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))   # Hidden layer 2: 1024 -> 512
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))   # Hidden layer 3: 512 -> 256
        x = self.fc4(x)                      # Output layer: 256 -> 10 logits
        return x

    def prunable_layers(self):
        # Helper to retrieve all PrunableLinear layers for sparsity calculations
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]


# Part 2b: Sparsity Loss
# Two-term loss:
# Term 1: mean(gates)
#   Constant gradient regardless of gate magnitude, drives gates all the way to 0.
#   Unlike L2 whose gradient shrinks near 0, leaving gates small but never zero.
# Term 2: mean(gate * (1 - gate))
#   Maximized at gate=0.5 and minimized at gate=0 or gate=1.
#   Minimizing it forces gates to commit to either 0 (pruned) or 1 (kept),
#   which creates the bimodal distribution: spike at 0, cluster near 1.
# Both terms are normalized by number of layers to stay on same scale as CrossEntropy.

def sparsity_loss(model):
    l1_total      = 0.0
    bimodal_total = 0.0
    count         = 0
    for layer in model.prunable_layers():
        gates          = torch.sigmoid(layer.gate_scores)
        l1_total      += gates.mean()
        bimodal_total += (gates * (1.0 - gates)).mean()
        count         += 1
    return (l1_total + bimodal_total) / count


def compute_sparsity(model, threshold=1e-2):
    # Returns what percentage of gates are below the pruning threshold.
    # A gate below threshold means the corresponding weight is effectively pruned.
    total  = 0
    pruned = 0
    for layer in model.prunable_layers():
        g       = layer.get_gates()
        total  += g.numel()
        pruned += (g < threshold).sum().item()
    return 100.0 * pruned / total


# Dataset
# Training set uses random flip and crop augmentation.
# Test set uses only normalization for fair evaluation.

def get_dataloaders(batch_size=256):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    train_set    = datasets.CIFAR10('./data', train=True,  download=True, transform=transform_train)
    test_set     = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    pin          = (device == 'cuda')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    return train_loader, test_loader


train_loader, test_loader = get_dataloaders(batch_size=256)
print("Train samples:", len(train_loader.dataset))
print("Test  samples:", len(test_loader.dataset))


# Part 3: Evaluation

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y     = x.to(device), y.to(device)
            pred      = model(x).argmax(1)
            correct  += pred.eq(y).sum().item()
            total    += y.size(0)
    return 100.0 * correct / total


# Part 3: Training Loop
# Total Loss = CrossEntropyLoss + lambda * SparsityLoss
# gate_scores use 10x learning rate so they respond faster to the sparsity signal.
# CosineAnnealingLR gradually reduces the learning rate for smooth convergence.
# Gradient clipping prevents exploding gradients at high lambda values.

def train_model(lam, epochs=100, lr=1e-3):
    print("Lambda:", lam, "| Epochs:", epochs, "| LR:", lr)

    model = SelfPruningNet().to(device)

    # Separate parameter groups: gate_scores learn faster (10x lr)
    gate_params  = [p for n, p in model.named_parameters() if 'gate_scores' in n]
    other_params = [p for n, p in model.named_parameters() if 'gate_scores' not in n]
    optimizer    = optim.Adam([
        {'params': other_params, 'lr': lr},
        {'params': gate_params,  'lr': lr * 10}
    ])

    # CosineAnnealingLR smoothly reduces lr to near-zero by the final epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ce        = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = run_cls = run_spar = correct = total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out       = model(x)
            cls_loss  = ce(out, y)
            spar_loss = sparsity_loss(model)
            loss      = cls_loss + lam * spar_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            run_loss += loss.item()      * x.size(0)
            run_cls  += cls_loss.item()  * x.size(0)
            run_spar += spar_loss.item() * x.size(0)
            pred      = out.argmax(1)
            correct  += pred.eq(y).sum().item()
            total    += y.size(0)

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            spar_pct  = compute_sparsity(model)
            gate_mean = sparsity_loss(model).item()
            print("Ep", str(epoch).zfill(2) + "/" + str(epochs),
                  "| Loss:", round(run_loss/total, 4),
                  "| CE:", round(run_cls/total, 4),
                  "| SparsLoss:", round(lam * run_spar/total, 4),
                  "| TrainAcc:", round(100*correct/total, 2),
                  "| Gates<0.01:", round(spar_pct, 2),
                  "| GateMean:", round(gate_mean, 4))

    test_acc = evaluate(model, test_loader)
    spar_lvl = compute_sparsity(model, threshold=1e-2)
    print("Test Accuracy:", round(test_acc, 2), "%  |  Sparsity Level:", round(spar_lvl, 2), "%")
    return test_acc, spar_lvl, model


# Run experiments for three lambda values
# Low    lambda (1.0)  -> mild pruning,   high accuracy,  ~30-50% sparsity
# Medium lambda (5.0)  -> balanced,       good accuracy,  ~60-75% sparsity
# High   lambda (20.0) -> aggressive,     slight accuracy drop, ~80-90% sparsity

LAMBDAS = [1.0, 5.0, 20.0]
EPOCHS  = 100
results = []
models  = {}

for lam in LAMBDAS:
    acc, spar, model = train_model(lam=lam, epochs=EPOCHS, lr=1e-3)
    results.append({'lambda': lam, 'accuracy': acc, 'sparsity': spar})
    models[lam] = model

# Final Results Table
print("Lambda | Test Accuracy | Sparsity (%)")
for r in results:
    print(r['lambda'], "|", round(r['accuracy'], 2), "% |", round(r['sparsity'], 2), "%")


# Gate Distribution Plot for ALL three models
# Each model gets its own row of two panels:
#   Left  -> full distribution showing the spike at 0 (pruned gates)
#   Right -> surviving gates only (the cluster away from 0)
# A successful result shows a bimodal pattern: spike at 0 and cluster at 0.5-1.

def plot_all_gate_distributions(models_dict, lambdas):
    n         = len(lambdas)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    fig.suptitle('Gate Value Distributions for All Lambda Values',
                 fontsize=15, fontweight='bold', y=1.01)

    for i, lam in enumerate(lambdas):
        model = models_dict[lam]

        all_gates = np.concatenate([
            layer.get_gates().cpu().numpy().flatten()
            for layer in model.prunable_layers()
        ])

        sparsity_pct = 100.0 * np.mean(all_gates < 0.01)
        surviving    = all_gates[all_gates >= 0.01]
        row_title    = 'lambda=' + str(lam) + '  |  Sparsity=' + str(round(sparsity_pct, 1)) + '%'

        # Left panel: full distribution (0 to 1)
        axes[i][0].hist(all_gates, bins=100, color='#01696f',
                        edgecolor='white', linewidth=0.3, alpha=0.85)
        axes[i][0].axvline(0.01, color='#a13544', linestyle='--',
                           linewidth=2, label='Prune threshold (0.01)')
        axes[i][0].set_xlabel('Gate Value [sigmoid(gate_score)]', fontsize=10)
        axes[i][0].set_ylabel('Count', fontsize=10)
        axes[i][0].set_title('Full Distribution  |  ' + row_title, fontsize=11)
        axes[i][0].legend(fontsize=9)
        axes[i][0].spines[['top', 'right']].set_visible(False)

        # Right panel: surviving gates only (cluster away from 0)
        if len(surviving) > 0:
            axes[i][1].hist(surviving, bins=60, color='#a13544',
                            edgecolor='white', linewidth=0.3, alpha=0.85)
            axes[i][1].set_title(
                'Surviving Gates  |  ' + str(len(surviving)) + ' active weights', fontsize=11
            )
        else:
            axes[i][1].text(0.5, 0.5, 'No surviving gates above threshold',
                            ha='center', va='center',
                            transform=axes[i][1].transAxes, fontsize=10)
            axes[i][1].set_title('Surviving Gates', fontsize=11)

        axes[i][1].set_xlabel('Gate Value (surviving gates only)', fontsize=10)
        axes[i][1].set_ylabel('Count', fontsize=10)
        axes[i][1].spines[['top', 'right']].set_visible(False)

        print("lambda =", lam,
              "| Total:", len(all_gates),
              "| Pruned:", int(len(all_gates) * sparsity_pct / 100),
              "(", round(sparsity_pct, 1), "%)",
              "| Surviving:", len(surviving),
              "(", round(100 - sparsity_pct, 1), "%)")

    plt.tight_layout()
    plt.savefig('gate_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: gate_distribution.png")


plot_all_gate_distributions(models, LAMBDAS)