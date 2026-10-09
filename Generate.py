import os
import random

def generate_dataset(filename="Data.txt", num_samples=10000, num_features=5):
    """
    Generates synthetic multivariate regression dataset with known coefficients and noise:
    y = b + w1*x1 + w2*x2 + ... + wn*xn + noise
    Saved in CSV format: x1,x2,x3,...,xn,y
    """
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(scriptdir, filename)

    random.seed(42)
    weights = [3.5, -2.0, 1.5, -0.8, 2.2]
    # Adjust weights length if num_features changes
    if len(weights) < num_features:
        weights += [round(random.uniform(-3, 3), 2) for _ in range(num_features - len(weights))]
    else:
        weights = weights[:num_features]
    bias = 4.0

    print(f"Generating {num_samples} samples with {num_features} features into {filepath}...")
    with open(filepath, "w") as f:
        for _ in range(num_samples):
            features = [round(random.gauss(0, 1), 4) for _ in range(num_features)]
            target = bias + sum(w * x for w, x in zip(weights, features)) + random.gauss(0, 0.25)
            row = features + [round(target, 4)]
            f.write(",".join(map(str, row)) + "\n")

    print(f"Dataset generated successfully! File size: {os.path.getsize(filepath) / 1024:.2f} KB")

if __name__ == "__main__":
    generate_dataset()