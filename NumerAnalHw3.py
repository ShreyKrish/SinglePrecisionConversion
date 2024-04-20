import numpy as np

def solve_system(A, b):
    x = np.linalg.solve(A, b)

    max_iterations = 100
    tolerance = 1e-6
    iteration = 0

    while iteration < max_iterations:
        r = b - np.dot(A, x)
        r_single_precision = np.array(r, dtype=np.float32)  # Convert to single-precision

        z = np.linalg.solve(A, r_single_precision)

        x = x + z

        if np.linalg.norm(z, np.inf) < tolerance:
            print(f"Converged in {iteration + 1} iterations.")
            break

        iteration += 1

    return x

if __name__ == "__main__":
    # Define matrix A and vector b
    A = np.array([[21.0, 67.0, 88.0, 73.0],
                  [76.0, 63.0, 7.0, 20.0],
                  [0.0, 85.0, 56.0, 54.0],
                  [19.3, 43.0, 30.2, 29.4]])

    b = np.array([141.0, 109.0, 218.0, 93.7])

    solution = solve_system(A, b)

    print("Solution x:", solution)
