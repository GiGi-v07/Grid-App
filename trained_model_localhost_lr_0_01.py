"""
Standalone Trained AI Model Script
Automatically generated after training on dataset.
Learned Parameters: w1=2.499037, w2=-1.499856, bias=2.939864 (lr=0.01)
Test Accuracy (R^2): 99.99% | Test MSE: 0.004116
"""
import sys

# Baked-in learned weights from training
W1 = 2.49903700
W2 = -1.49985600
BIAS = 2.93986400

# Embedded sample test records (x1, x2, actual_y) for self-verification
TEST_SAMPLES = [(1.2, -2.4, 9.62), (-3.1, 1.5, -7.02), (4.0, 0.8, 11.79), (-1.5, -3.2, 4.06), (2.75, 3.1, 5.21)]


def predict(x1, x2):
    """Predicts target output y using the trained model weights."""
    return (W1 * float(x1)) + (W2 * float(x2)) + BIAS


def run_self_test():
    """Runs self-verification on embedded test samples and supports CLI input."""
    print("=" * 60)
    print("         TRAINED AI MODEL - STANDALONE EXECUTION")
    print("=" * 60)
    print(f"Learned Equation : y = {W1:.4f}*x1 + ({W2:.4f})*x2 + {BIAS:.4f}")
    print(f"Training LR      : 0.01")
    print(f"Test R^2 Accuracy: 99.99%\n")

    print("--- Self-Test on Verification Data ---")
    for x1, x2, actual in TEST_SAMPLES:
        pred = predict(x1, x2)
        print(f"Input (x1={x1:6.2f}, x2={x2:6.2f}) -> Predicted: {pred:7.3f} | Actual: {actual:7.3f} | Error: {abs(pred - actual):.3f}")

    if len(sys.argv) >= 3:
        ux1, ux2 = float(sys.argv[1]), float(sys.argv[2])
        print(f"\nCustom Prediction for ({ux1}, {ux2}): {predict(ux1, ux2):.4f}")


if __name__ == "__main__":
    run_self_test()
