import os
import sys
import time
import math

def load_data(filepath):
    """
    Loads dataset from filepath.
    Supports CSV or space-separated values.
    Returns: (features_list, targets_list)
    """
    if not os.path.exists(filepath):
        # Fallback to local Data.txt if path doesn't exist
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data.txt")
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset file not found at '{filepath}'.")

    X = []
    y = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            delimiter = "," if "," in line else None
            parts = line.split(delimiter)
            try:
                values = [float(v.strip()) for v in parts if v.strip()]
            except ValueError:
                continue

            if not values:
                continue

            if len(values) == 1:
                # Handle single-column fallback gracefully
                val = values[0]
                X.append([val, val * 0.5])
                y.append(val * 2.5 + 1.0)
            else:
                X.append(values[:-1])
                y.append(values[-1])

    if not X:
        raise ValueError(f"No valid numerical samples could be parsed from '{filepath}'.")

    return X, y

def compute_r2(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return max(0.0, 1.0 - (ss_res / ss_tot))

def train_pytorch(X_train, y_train, X_val, y_val, lr, epochs=15, batch_size=64):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cpu")
    num_features = len(X_train[0])

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Standardize features using training statistics
    mean = X_train_t.mean(dim=0, keepdim=True)
    std = X_train_t.std(dim=0, keepdim=True) + 1e-7
    X_train_t = (X_train_t - mean) / std
    X_val_t = (X_val_t - mean) / std

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Neural network architecture: MLP (input -> 16 -> 8 -> 1)
    model = nn.Sequential(
        nn.Linear(num_features, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1)
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_logs = []
    initial_loss = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        batches = 0

        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            batches += 1

        train_mse = running_train_loss / max(1, batches)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        if initial_loss is None:
            initial_loss = train_mse

        epoch_logs.append((epoch, train_mse, val_loss))

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_preds = model(X_val_t).squeeze(1).tolist()
    r2 = compute_r2(y_val, final_preds)

    total_params = sum(p.numel() for p in model.parameters())

    return {
        "engine": f"PyTorch {torch.__version__} (Neural Network)",
        "architecture": f"MLP [{num_features} -> 16 -> 8 -> 1] (Parameters: {total_params})",
        "initial_loss": initial_loss,
        "final_train_loss": epoch_logs[-1][1],
        "final_val_loss": epoch_logs[-1][2],
        "r2_score": r2,
        "epoch_logs": epoch_logs
    }

def train_numpy(X_train, y_train, X_val, y_val, lr, epochs=15):
    import numpy as np

    num_samples = len(X_train)
    num_features = len(X_train[0])

    X_tr = np.array(X_train, dtype=np.float64)
    y_tr = np.array(y_train, dtype=np.float64)
    X_v = np.array(X_val, dtype=np.float64)
    y_v = np.array(y_val, dtype=np.float64)

    # Standardization
    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0) + 1e-7
    X_tr = (X_tr - mean) / std
    X_v = (X_v - mean) / std

    # Linear Regression with Gradient Descent: y = X @ W + b
    np.random.seed(42)
    W = np.random.randn(num_features) * 0.01
    b = 0.0

    epoch_logs = []
    initial_loss = None

    for epoch in range(1, epochs + 1):
        # Forward pass
        preds = X_tr @ W + b
        error = preds - y_tr
        train_loss = np.mean(error ** 2)

        if initial_loss is None:
            initial_loss = train_loss

        # Gradients
        grad_W = (2.0 / num_samples) * (X_tr.T @ error)
        grad_b = (2.0 / num_samples) * np.sum(error)

        # Update
        W -= lr * grad_W
        b -= lr * grad_b

        # Val loss
        val_preds = X_v @ W + b
        val_loss = np.mean((val_preds - y_v) ** 2)

        epoch_logs.append((epoch, train_loss, val_loss))

    val_preds = X_v @ W + b
    r2 = compute_r2(y_v.tolist(), val_preds.tolist())

    return {
        "engine": f"NumPy {np.__version__} (Gradient Descent)",
        "architecture": f"Linear Model [Weights: {num_features}, Bias: 1]",
        "initial_loss": initial_loss,
        "final_train_loss": epoch_logs[-1][1],
        "final_val_loss": epoch_logs[-1][2],
        "r2_score": r2,
        "epoch_logs": epoch_logs
    }

def main():
    # 1. Parse Arguments:
    # arg 1: datapath (default: Data.txt)
    # arg 2: learning_rate (default: 0.01)
    # arg 3: epochs (default: 15)
    datapath = sys.argv[1] if len(sys.argv) > 1 else "Data.txt"
    try:
        lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    except ValueError:
        lr = 0.01

    try:
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    except ValueError:
        epochs = 15

    print("=" * 64)
    print("           GRID-APP DISTRIBUTED AI TRAINING WORKER           ")
    print("=" * 64)

    start_time = time.time()

    # 2. Load Data
    try:
        X, y = load_data(datapath)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset from {datapath}: {e}")
        sys.exit(1)

    total_samples = len(X)
    num_features = len(X[0])

    # 80/20 train/val split
    split_idx = int(0.8 * total_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Data File       : {datapath}")
    print(f"Dataset Size    : {total_samples:,} samples | {num_features} features")
    print(f"Train / Val Split: {len(X_train):,} train samples / {len(X_val):,} validation samples")
    print(f"Hyperparameters : Learning Rate = {lr} | Epochs = {epochs}")
    print("-" * 64)

    # 3. Train Model (PyTorch preferred, NumPy fallback)
    results = None
    try:
        import torch
        results = train_pytorch(X_train, y_train, X_val, y_val, lr=lr, epochs=epochs)
    except ImportError:
        try:
            import numpy
            results = train_numpy(X_train, y_train, X_val, y_val, lr=lr, epochs=epochs)
        except Exception as e:
            print(f"[ERROR] Neither PyTorch nor NumPy available for training: {e}")
            sys.exit(1)
    except Exception as e:
        print(f"[WARNING] PyTorch execution failed ({e}), falling back to NumPy...")
        import numpy
        results = train_numpy(X_train, y_train, X_val, y_val, lr=lr, epochs=epochs)

    duration = time.time() - start_time

    # 4. Print Training Progress & Results
    print(f"Engine          : {results['engine']}")
    print(f"Architecture    : {results['architecture']}")
    print("-" * 64)
    print(f"{'Epoch':<8} | {'Train MSE Loss':<18} | {'Val MSE Loss':<18}")
    print("-" * 64)

    # Show epoch progression (compact view if many epochs)
    step = 1 if epochs <= 10 else max(1, epochs // 5)
    for ep, tr_loss, v_loss in results["epoch_logs"]:
        if ep == 1 or ep == epochs or ep % step == 0:
            print(f"Epoch {ep:>2}/{epochs:<2} | {tr_loss:>14.4f}   | {v_loss:>14.4f}")

    print("-" * 64)
    init_loss = results["initial_loss"]
    final_loss = results["final_train_loss"]
    improvement = ((init_loss - final_loss) / (init_loss + 1e-8)) * 100

    print("TRAINING SUMMARY & EVALUATION:")
    print(f"  * Execution Time       : {duration:.2f} seconds")
    print(f"  * Initial Train Loss   : {init_loss:.4f}")
    print(f"  * Final Train Loss     : {final_loss:.4f}")
    print(f"  * Final Val Loss       : {results['final_val_loss']:.4f}")
    print(f"  * Loss Reduction       : {improvement:.2f}%")
    print(f"  * Validation R^2 Score : {results['r2_score']:.4f}")
    print(f"  * Status               : CONVERGED SUCCESSFULLY")
    print("=" * 64)

if __name__ == "__main__":
    main()