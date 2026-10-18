import sys
import os

def load_data(filepath):
    x_data = []
    y_data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(val) for val in line.split(",")]
            x_data.append(parts[:-1])
            y_data.append(parts[-1])
    return x_data, y_data


def train():
    # Read arguments: sys.argv[1] = datapath, sys.argv[2] = learning_rate
    scriptdir = os.path.dirname(__file__) if "__file__" in globals() else "."
    default_data = os.path.join(scriptdir, "ai_data.txt")
    
    datapath = sys.argv[1] if len(sys.argv) > 1 else default_data
    try:
        learning_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    except ValueError:
        learning_rate = 0.01

    print(f"--- Tiny AI Training Job ---")
    print(f"Dataset path: {datapath}")
    print(f"Learning rate: {learning_rate}")

    X, Y = load_data(datapath)
    n = len(Y)
    if n == 0:
        print("Error: Empty dataset.")
        return

    # Model parameters: y_hat = w1 * x1 + w2 * x2 + bias
    w1, w2, bias = 0.0, 0.0, 0.0
    epochs = 50

    for epoch in range(1, epochs + 1):
        # Forward pass & loss
        loss = 0.0
        dw1, dw2, dbias = 0.0, 0.0, 0.0

        for (x1, x2), y in zip(X, Y):
            y_hat = w1 * x1 + w2 * x2 + bias
            err = y_hat - y
            loss += err ** 2
            dw1 += 2 * err * x1
            dw2 += 2 * err * x2
            dbias += 2 * err

        loss /= n
        dw1 /= n
        dw2 /= n
        dbias /= n

        # Gradient descent update
        w1 -= learning_rate * dw1
        w2 -= learning_rate * dw2
        bias -= learning_rate * dbias

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Loss: {loss:.4f} | w1: {w1:.3f}, w2: {w2:.3f}, b: {bias:.3f}")

    print(f"Training Complete! Final Loss: {loss:.4f}")
    print(f"Learned Equation: y = {w1:.3f}*x1 + {w2:.3f}*x2 + {bias:.3f}")


if __name__ == "__main__":
    train()

