from types import Value


if __name__ == "__main__":
    if __name__ == "__main__":
        print("--- Running Forward Pass ---")

        # 1. Define our initial inputs/weights
        a = Value(2.0)
        b = Value(-3.0)
        d = Value(4.0)

        # 2. Compute a dynamic expression: c = a * b * d
        # Python executes this as:
        # step1 = a * b
        # c = step1 * d
        c = a * b * d

        print(f"Input a: {a.data}")
        print(f"Input b: {b.data}")
        print(f"Input d: {d.data}")
        print(f"Output c (result of a * b * d): {c.data}\n")

        print("--- Running Backward Pass ---")
        # 3. Call backward on the final output node
        c.backward()

        print("--- Calculated Gradients (Sensitivities) ---")
        print(f"dc/dc (Should be 1.0): {c.grad}")
        print(f"dc/dd (Should be a*b = -6.0): {d.grad}")
        print(f"dc/db (Should be a*d = 8.0): {b.grad}")
        print(f"dc/da (Should be b*d = -12.0): {a.grad}")
