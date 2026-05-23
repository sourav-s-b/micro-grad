from types import Value

if __name__ == "__main__":
    print("--- Simulating One Neuron with Tanh Activation ---")

    # Inputs (x1, x2)
    x1 = Value(2.0)
    x2 = Value(0.0)

    # Weights (w1, w2) - these represent what the network learns
    w1 = Value(-3.0)
    w2 = Value(1.0)

    # Bias of the neuron (b)
    b = Value(6.881373587019543)  # Specific number chosen for clean outputs

    # Step 1: Multiply inputs by weights and sum them up
    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1_plus_x2w2 = x1w1 + x2w2

    # Step 2: Add the bias (Pre-activation output)
    n = x1w1_plus_x2w2 + b

    # Step 3: Pass through our brand new activation function (Post-activation output)
    output = n.tanh()

    # Run backpropagation
    output.backward()

    print(f"Neuron Raw Sum (n): {n.data:.4f}")
    print(f"Squashed Output   : {output.data:.4f}\n")

    print("--- Gradients Flowed All the Way Back ---")
    print(f"Weight 1 Gradient (dOut/dw1): {w1.grad:.4f} (Should be ~0.7000)")
    print(f"Weight 2 Gradient (dOut/dw2): {w2.grad:.4f} (Should be ~0.0000)")
    print(f"Bias Gradient     (dOut/db) : {b.grad:.4f} (Should be ~0.5000)")
