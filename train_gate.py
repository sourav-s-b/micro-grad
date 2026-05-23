from core.nn import MLP

# 1. Create a dataset for an XOR Gate
# Inputs: [x1, x2] -> Output: Expected y
xs = [
    [2.0, 3.0],   # Class 1
    [3.0, -1.0],  # Class 2
    [-2.0, 5.0],  # Class 1
    [1.0, 1.0],   # Class 2
]
ys = [1.0, -1.0, 1.0, -1.0] # Labels

# 2. Initialize a 2-input network with a hidden layer of 4 neurons, and 1 output neuron
model = MLP(2, [4, 4, 1])
print(f"Initialized vanguard_grad network with {len(model.parameters())} parameters.")

# 3. Optimization Loop
epochs = 50
learning_rate = 0.05

print("\n--- Kicking off Training Optimization Loop ---")
for k in range(epochs):
    
    # Forward Pass: get predictions for all samples
    ypred = [model(x) for x in xs]
    
    # Calculate Mean Squared Error (MSE) Loss
    loss = sum((ypot - ygt)**2 for ygt, ypot in zip(ys, ypred))
    
    # Backward Pass: Reset old gradients and calculate the new ones
    model.zero_grad()
    loss.backward()
    
    # Update weights and biases (Gradient Descent)
    for p in model.parameters():
        p.data -= learning_rate * p.grad
        
    if k % 5 == 0 or k == epochs - 1:
        print(f"Epoch {k:02d} | Total Loss: {loss.data:.6f}")

print("\n--- Final Model Predictions After Training ---")
final_preds = [model(x) for x in xs]
for x, target, pred in zip(xs, ys, final_preds):
    print(f"Input: {x} -> Target: {target:2.1f} | Model Predicted: {pred.data: .4f}")