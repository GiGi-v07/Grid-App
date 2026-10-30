import sys
import os
import json
import math

# ==============================================================================
# DEFAULT CONFIGURATION
# These values are used when the script is run without CLI arguments or when
# overridden dynamically by AITrainer.py before uploading to a worker node.
# ==============================================================================
DEFAULT_DATA_FILE = "ai_data.txt"
DEFAULT_TEST_FILE = "ai_test_data.txt"
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPOCHS = 200
SAVE_LOCAL_ARTIFACTS = True


def load_data(filepath):
    """
    Reads a comma-separated dataset file (x1, x2, y) and returns:
    - x_data: list of feature pairs [x1, x2]
    - y_data: list of target values y
    Ignores blank lines and comment lines starting with '#'.
    """
    x_data = []
    y_data = []
    if not os.path.exists(filepath):
        return x_data, y_data

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(val) for val in line.split(",")]
            if len(parts) >= 3:
                x_data.append(parts[:2])
                y_data.append(parts[2])
    return x_data, y_data


def evaluate_model(x_test, y_test, w1, w2, bias):
    """
    Tests the trained linear regression model (y_hat = w1*x1 + w2*x2 + bias)
    against unseen testing data.
    Returns a dictionary containing:
    - mse: Mean Squared Error on the test set
    - mae: Mean Absolute Error on the test set
    - r2_score: R-squared accuracy score (1.0 = 100% perfect fit)
    - samples: List of sample predictions vs actual values
    """
    n = len(y_test)
    if n == 0:
        return {"mse": 0.0, "mae": 0.0, "r2_score": 0.0, "samples": []}

    mse = 0.0
    mae = 0.0
    y_mean = sum(y_test) / n
    ss_tot = 0.0
    ss_res = 0.0
    samples = []

    for i, ((x1, x2), y_true) in enumerate(zip(x_test, y_test)):
        y_pred = w1 * x1 + w2 * x2 + bias
        err = y_pred - y_true
        mse += err ** 2
        mae += abs(err)
        ss_res += err ** 2
        ss_tot += (y_true - y_mean) ** 2
        if i < 5:
            samples.append({
                "x1": round(x1, 2),
                "x2": round(x2, 2),
                "actual": round(y_true, 3),
                "predicted": round(y_pred, 3),
                "error": round(abs(err), 3),
            })

    mse /= n
    mae /= n
    r2_score = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "r2_score": r2_score,
        "samples": samples,
    }


def generate_standalone_model_script(output_path, weights_info, test_samples):
    """
    Generates a standalone, working Python script with the learned model weights
    baked directly into the code so it can be executed independently.
    """
    w1 = weights_info["w1"]
    w2 = weights_info["w2"]
    bias = weights_info["bias"]
    lr = weights_info["learning_rate"]
    r2_pct = max(0.0, weights_info["test_r2"] * 100.0)

    script_content = f'''"""
Standalone Trained AI Model Script
Automatically generated after training on dataset.
Learned Parameters: w1={w1:.6f}, w2={w2:.6f}, bias={bias:.6f} (lr={lr})
Test Accuracy (R^2): {r2_pct:.2f}% | Test MSE: {weights_info["test_mse"]:.6f}
"""
import sys

# Baked-in learned weights from training
W1 = {w1:.8f}
W2 = {w2:.8f}
BIAS = {bias:.8f}

# Embedded sample test records (x1, x2, actual_y) for self-verification
TEST_SAMPLES = {repr([(s["x1"], s["x2"], s["actual"]) for s in test_samples])}


def predict(x1, x2):
    """Predicts target output y using the trained model weights."""
    return (W1 * float(x1)) + (W2 * float(x2)) + BIAS


def run_self_test():
    """Runs self-verification on embedded test samples and supports CLI input."""
    print("=" * 60)
    print("         TRAINED AI MODEL - STANDALONE EXECUTION")
    print("=" * 60)
    print(f"Learned Equation : y = {{W1:.4f}}*x1 + ({{W2:.4f}})*x2 + {{BIAS:.4f}}")
    print(f"Training LR      : {lr}")
    print(f"Test R^2 Accuracy: {r2_pct:.2f}%\\n")

    print("--- Self-Test on Verification Data ---")
    for x1, x2, actual in TEST_SAMPLES:
        pred = predict(x1, x2)
        print(f"Input (x1={{x1:6.2f}}, x2={{x2:6.2f}}) -> Predicted: {{pred:7.3f}} | Actual: {{actual:7.3f}} | Error: {{abs(pred - actual):.3f}}")

    if len(sys.argv) >= 3:
        ux1, ux2 = float(sys.argv[1]), float(sys.argv[2])
        print(f"\\nCustom Prediction for ({{ux1}}, {{ux2}}): {{predict(ux1, ux2):.4f}}")


if __name__ == "__main__":
    run_self_test()
'''
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script_content)


def train():
    """
    Main training and evaluation routine:
    1. Loads training and testing data (splitting 80/20 if no separate test file is present).
    2. Trains a multi-feature linear regression model via Batch Gradient Descent.
    3. Evaluates the trained model on unseen testing data.
    4. Saves JSON model weights and a standalone runnable Python script.
    5. Emits a structured JSON summary line for distributed aggregation.
    """
    scriptdir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    default_data = os.path.join(scriptdir, DEFAULT_DATA_FILE)
    default_test = os.path.join(scriptdir, DEFAULT_TEST_FILE)
    if not os.path.exists(default_test):
        parent_test = os.path.join(os.path.dirname(scriptdir), DEFAULT_TEST_FILE)
        if os.path.exists(parent_test):
            default_test = parent_test

    datapath = sys.argv[1] if len(sys.argv) > 1 else default_data
    try:
        learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else float(DEFAULT_LEARNING_RATE)
    except ValueError:
        learning_rate = float(DEFAULT_LEARNING_RATE)

    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else int(DEFAULT_EPOCHS)

    print("=== AI Model Training & Evaluation ===")
    print(f"Dataset Path  : {os.path.basename(datapath)}")
    print(f"Learning Rate : {learning_rate}")
    print(f"Epochs        : {epochs}")

    all_x, all_y = load_data(datapath)
    if len(all_y) == 0:
        print("Error: Dataset is empty or could not be loaded.")
        return

    # Load dedicated test file if available locally; otherwise split 80% train / 20% test
    test_x, test_y = load_data(default_test)
    if len(test_y) > 0 and os.path.abspath(datapath) != os.path.abspath(default_test):
        train_x, train_y = all_x, all_y
    else:
        split_idx = max(1, int(len(all_y) * 0.8))
        train_x, train_y = all_x[:split_idx], all_y[:split_idx]
        test_x, test_y = all_x[split_idx:], all_y[split_idx:]
        if len(test_y) == 0:
            test_x, test_y = train_x, train_y

    n_train = len(train_y)
    n_test = len(test_y)
    print(f"Train Samples : {n_train} | Test Samples: {n_test}\n")

    # Initialize model parameters: y_hat = w1 * x1 + w2 * x2 + bias
    w1, w2, bias = 0.0, 0.0, 0.0
    train_loss = 0.0
    diverged = False

    for epoch in range(1, epochs + 1):
        loss = 0.0
        dw1, dw2, dbias = 0.0, 0.0, 0.0

        for (x1, x2), y in zip(train_x, train_y):
            y_hat = w1 * x1 + w2 * x2 + bias
            err = y_hat - y
            loss += err ** 2
            dw1 += (2.0 * err * x1)
            dw2 += (2.0 * err * x2)
            dbias += (2.0 * err)

        train_loss = loss / n_train
        dw1 /= n_train
        dw2 /= n_train
        dbias /= n_train

        # Gradient descent parameter update
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        bias -= learning_rate * dbias

        if math.isnan(train_loss) or math.isinf(train_loss) or train_loss > 1e12:
            print(f"[!] Training diverged at epoch {epoch} (learning rate {learning_rate} is too high).")
            diverged = True
            break

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs:3d} | Train MSE: {train_loss:9.4f} | w1: {w1:6.3f}, w2: {w2:6.3f}, bias: {bias:6.3f}")

    # Evaluate the trained model on testing data
    eval_metrics = evaluate_model(test_x, test_y, w1, w2, bias)
    r2_pct = max(0.0, eval_metrics["r2_score"] * 100.0) if not diverged else 0.0

    print("\n--- Testing Data Evaluation ---")
    print(f"Learned Equation : y = {w1:.4f}*x1 + ({w2:.4f})*x2 + {bias:.4f}")
    print(f"Test Set MSE     : {eval_metrics['mse']:.4f}")
    print(f"Test Set MAE     : {eval_metrics['mae']:.4f}")
    print(f"Test Accuracy R2 : {r2_pct:.2f}%")

    if eval_metrics["samples"]:
        print("\nSample Test Predictions vs Actual:")
        for s in eval_metrics["samples"]:
            print(f"  x=({s['x1']:5.2f}, {s['x2']:5.2f}) -> Predicted: {s['predicted']:7.3f} | Actual: {s['actual']:7.3f} | Error: {s['error']:.3f}")

    result_payload = {
        "w1": round(w1, 6) if not diverged else 0.0,
        "w2": round(w2, 6) if not diverged else 0.0,
        "bias": round(bias, 6) if not diverged else 0.0,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "train_samples": n_train,
        "test_samples": n_test,
        "train_mse": round(train_loss, 6) if not (diverged or math.isnan(train_loss) or math.isinf(train_loss)) else 999999.0,
        "test_mse": round(eval_metrics["mse"], 6) if not (diverged or math.isnan(eval_metrics["mse"]) or math.isinf(eval_metrics["mse"])) else 999999.0,
        "test_mae": round(eval_metrics["mae"], 6) if not (diverged or math.isnan(eval_metrics["mae"]) or math.isinf(eval_metrics["mae"])) else 999999.0,
        "test_r2": round(eval_metrics["r2_score"], 6) if not (diverged or math.isnan(eval_metrics["r2_score"]) or math.isinf(eval_metrics["r2_score"])) else 0.0,
        "diverged": diverged,
    }

    # Clean up temporary uploaded dataset slice on worker nodes after training
    if os.path.basename(datapath).startswith("temp_data_") and os.path.exists(datapath):
        try:
            os.remove(datapath)
        except OSError:
            pass

    # Save local JSON weights and standalone runnable model inside Trained_Models/ when run directly
    if SAVE_LOCAL_ARTIFACTS:
        try:
            models_dir = os.path.join(scriptdir, "Trained_Models")
            os.makedirs(models_dir, exist_ok=True)
            weights_file = os.path.join(models_dir, "trained_weights.json")
            with open(weights_file, "w", encoding="utf-8") as wf:
                json.dump(result_payload, wf, indent=4)
            model_script = os.path.join(models_dir, "trained_model.py")
            generate_standalone_model_script(model_script, result_payload, eval_metrics["samples"])
            print(f"\n[Saved] JSON Weights     : Trained_Models/{os.path.basename(weights_file)}")
            print(f"[Saved] Runnable Script  : Trained_Models/{os.path.basename(model_script)}")
        except Exception as e:
            print(f"Warning: Could not save local model files: {e}")

    # Emit machine-readable JSON line for distributed collection by AITrainer.py
    print(f"\nMODEL_RESULT_JSON:{json.dumps(result_payload)}")


if __name__ == "__main__":
    train()
