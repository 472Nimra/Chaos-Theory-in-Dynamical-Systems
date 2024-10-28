# Chaotic Systems: Logistic Map and Lorenz System

## Project Overview

This project implements two classical examples of chaotic systems: the **Logistic Map** and the **Lorenz System**. Both systems exhibit sensitive dependence on initial conditions, a hallmark of chaotic behavior. The project generates and visualizes the bifurcation diagram and Lyapunov exponent for the Logistic Map, as well as the Lorenz attractor.

## Key Concepts

### Logistic Map

The **Logistic Map** is a polynomial mapping that is often used to model population growth. It is defined by the recurrence relation:

\[ x_{n+1} = r \cdot x_n \cdot (1 - x_n) \]

Where:
- \( x_n \) is the population at time step \( n \) (normalized between 0 and 1).
- \( r \) is a parameter that affects the growth rate.

As \( r \) varies, the system can exhibit stable points, periodic behavior, and chaotic dynamics. The bifurcation diagram shows how the behavior of the system changes as \( r \) varies.

### Bifurcation Diagram

A **bifurcation diagram** is a visual representation of the different states of a dynamical system as a parameter changes. In this case, it shows the values of \( x \) (population) for different values of \( r \).

### Lyapunov Exponent

The **Lyapunov exponent** quantifies the average rate of separation of infinitesimally close trajectories in a dynamical system. A positive Lyapunov exponent indicates chaos, while a negative exponent indicates stability. For the Logistic Map, the Lyapunov exponent is calculated for varying values of \( r \).

### Lorenz System

The **Lorenz system** is a system of ordinary differential equations originally developed to model atmospheric convection. It is defined by:

\[
\begin{align*}
\frac{dx}{dt} &= \sigma (y - x) \\
\frac{dy}{dt} &= x (\rho - z) - y \\
\frac{dz}{dt} &= xy - \beta z
\end{align*}
\]

Where \( \sigma \), \( \rho \), and \( \beta \) are system parameters. The solution to the Lorenz equations produces a three-dimensional attractor known as the **Lorenz attractor**, which exhibits chaotic behavior.

## Code Explanation

### Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import jit
```

- **NumPy** is used for numerical operations and array manipulation.
- **Matplotlib** is used for plotting results.
- **SciPy** is used for integrating ordinary differential equations.
- **Numba** is a just-in-time compiler that can optimize performance for numerical functions, though it's not explicitly used in the provided code.

### Logistic Map Function

```python
def logistic_map(r, x, n_iter=1000):
    """Generates logistic map values for given r and initial x."""
    x_vals = np.zeros(n_iter)
    x_vals[0] = x
    for i in range(1, n_iter):
        x_vals[i] = r * x_vals[i - 1] * (1 - x_vals[i - 1])
    return x_vals
```

- **Parameters**:
  - `r`: Growth rate parameter.
  - `x`: Initial population (0 < x < 1).
  - `n_iter`: Number of iterations (default 1000).
  
- The function initializes an array `x_vals` and calculates the next population value based on the Logistic Map formula for the specified number of iterations.

### Bifurcation Diagram

```python
def plot_bifurcation(r_min=2.5, r_max=4.0, n_iter=1000, last=100):
    """Generates and plots the bifurcation diagram for the logistic map."""
    r_values = np.linspace(r_min, r_max, 10000)
    x0 = 0.5  # Starting x value

    plt.figure(figsize=(10, 7))
    for r in r_values:
        x_vals = logistic_map(r, x0, n_iter)
        plt.plot([r] * last, x_vals[-last:], ',k', alpha=0.1)

    plt.title("Bifurcation Diagram of the Logistic Map")
    plt.xlabel("Parameter r")
    plt.ylabel("Population")
    plt.show()
```

- **Parameters**:
  - `r_min` and `r_max`: Range of the growth parameter \( r \).
  - `n_iter`: Number of iterations to calculate.
  - `last`: Number of last iterations to display in the plot.

- This function generates the bifurcation diagram by varying \( r \) from `r_min` to `r_max` and plotting the last 100 iterations for each \( r \).

### Lyapunov Exponent Calculation

```python
def lyapunov_exponent_logistic(r, n_iter=1000, x0=0.5):
    """Calculates the Lyapunov exponent for the logistic map."""
    x = x0
    lyapunov_sum = 0
    for _ in range(n_iter):
        x = r * x * (1 - x)
        lyapunov_sum += np.log(abs(r * (1 - 2 * x)))
    return lyapunov_sum / n_iter
```

- This function calculates the Lyapunov exponent for the Logistic Map by iterating \( n_iter \) times, updating the population, and summing the logarithmic derivatives.

### Lyapunov Exponent Plot

```python
def plot_lyapunov(r_min=2.5, r_max=4.0, n_iter=1000):
    """Plots Lyapunov exponent values for a range of r values."""
    r_values = np.linspace(r_min, r_max, 1000)
    lyapunov_values = [lyapunov_exponent_logistic(r, n_iter) for r in r_values]

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, lyapunov_values, 'b-', lw=0.5)
    plt.axhline(0, color='red', lw=0.8)
    plt.title("Lyapunov Exponent of the Logistic Map")
    plt.xlabel("Parameter r")
    plt.ylabel("Lyapunov Exponent")
    plt.show()
```

- This function generates a plot of the Lyapunov exponent as a function of \( r \), showing regions of chaos and stability.

### Lorenz System Definition

```python
def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
    """Defines the Lorenz system of differential equations."""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]
```

- This function defines the three differential equations that govern the Lorenz system's behavior.

### Plotting the Lorenz Attractor

```python
def plot_lorenz(x0=1.0, y0=1.0, z0=1.0, t_max=40, dt=0.01):
    """Simulates and plots the Lorenz attractor."""
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    initial_state = [x0, y0, z0]

    sol = solve_ivp(lorenz_system, t_span, initial_state, t_eval=t_eval, method='RK45')
    x, y, z = sol.y

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, lw=0.5)
    ax.set_title("Lorenz Attractor")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    plt.show()
```

- This function uses the `solve_ivp` function from SciPy to integrate the Lorenz system over a specified time span. It then visualizes the resulting trajectory in 3D.

### Main Script

```python
# Generate plots
plot_bifurcation()             # Plot bifurcation diagram for logistic map
plot_lyapunov()                # Plot Lyapunov exponents for logistic map
plot_lorenz()                  # Plot Lorenz attractor
```

- The main script calls the plotting functions to visualize the bifurcation diagram, Lyapunov exponents, and the Lorenz attractor.

## Conclusion

This project effectively demonstrates the principles of chaos theory through the Logistic Map and the Lorenz System. The bifurcation diagram and Lyapunov exponent reveal the transition from stable to chaotic behavior in the Logistic Map, while the Lorenz attractor illustrates how simple deterministic equations can lead to complex, unpredictable motion in three dimensions. This work serves as a valuable educational tool for understanding chaos in dynamical systems.
