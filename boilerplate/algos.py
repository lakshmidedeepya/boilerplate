from typing import Callable, Literal
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import numpy as np
import os

def conjugate_descent(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    approach: Literal["Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves"],
) -> NDArray[np.float64]:
    x = initial_point.copy()
    grad = d_f(x)
    d = -grad
    prev_grad = grad.copy()
    tol = 1e-6
    max_iter = 1000
    n = len(x)
    f_vals, x_history, grad_norms = [], [], []
    for k in range(max_iter):
        f_vals.append(f(x))
        x_history.append(x.copy())
        grad = d_f(x)
        grad_norms.append(np.linalg.norm(grad))
        if np.linalg.norm(grad) <= tol:
            break
        phi = lambda alpha: f(x + alpha * d)
        d_phi = lambda alpha: np.dot(d_f(x + alpha * d), d)

        alpha = bisection_line_search(phi, d_phi)
        x_new = x + (alpha * d)
        prev_grad = grad.copy()
        grad = d_f(x_new)


        if (k + 1) % n == 0:
            d = -grad
        else:
            if approach == "Fletcher-Reeves":
                beta = np.dot(grad, grad) / (np.dot(prev_grad, prev_grad) + 1e-12)
            elif approach == "Polak-Ribiere":
                beta = np.dot(grad, grad - prev_grad) / (np.dot(prev_grad, prev_grad) + 1e-12)
            elif approach == "Hestenes-Stiefel":
                beta = np.dot(grad, grad - prev_grad) / (np.dot(d, grad - prev_grad) + 1e-12)
            d = -grad + beta * d

        x = x_new
    
    plot_results(f, initial_point, approach, f_vals, grad_norms, x_history)
    return x


def sr1(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    x = initial_point.copy()
    n = len(x)
    B = np.eye(n)
    tol = 1e-8
    max_iter = 1000
    f_vals = []
    x_history = []
    grad_norms = []
    for k in range(max_iter):
        try:
            current_f = f(x)
            current_f=np.clip(current_f,-1e10, 1e10)
            grad = d_f(x)
            # Prevent overflow by clipping extreme values
            grad = np.clip(grad, -1e10, 1e10)
        except Exception:
            break

        f_vals.append(current_f)
        
        x_history.append(x.copy())
        grad_norms.append(np.linalg.norm(grad))
        if np.linalg.norm(grad) <= tol:
            break
        
        d = -B @ grad
        phi = lambda alpha: f(x + alpha * d)
        d_phi = lambda alpha: np.dot(d_f(x + alpha * d), d)
        alpha = bisection_line_search(phi, d_phi)

        s = alpha * d
        x_new = x + s
        
        try:
            
            y = d_f(x_new) - grad
            
            By= B @ y
           
            s_minus_By=s-By
            s_minus_By = np.clip(s_minus_By, -1e10, 1e10)
            
        except Exception:
            break

        denom = s_minus_By @ y

        if denom!=0 and abs(denom) >= tol * np.linalg.norm(s) * np.linalg.norm(s_minus_By):
            B += np.outer(s_minus_By, s_minus_By) / denom
            B = np.clip(B, -1e10, 1e10)

        x = x_new
        
    plot_results(f, initial_point, "sr1", f_vals, grad_norms, x_history)
    return x


def dfp(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    x = initial_point.copy()
    n = len(x)
    H = np.eye(n)
    tol = 1e-6
    max_iter = 9000
    f_vals = []
    x_history = []
    grad_norms = []
    for _ in range(max_iter):
        f_vals.append(f(x))
        x_history.append(x.copy())
        grad = d_f(x)
        grad_norms.append(np.linalg.norm(grad))
        if np.linalg.norm(grad) <= tol:
            break

        d = -H @ grad

        phi = lambda a: f(x + a * d)
        d_phi = lambda a: np.dot(d_f(x + a * d), d)
        if d_phi(0) >= 0:
            d = -grad
        alpha = bisection_line_search(phi, d_phi)

        s = alpha * d
        x_new = x + s
        y = d_f(x_new) - grad

        Hy = H @ y
        yH = y @ H
        denom = np.dot(y, s)
        if abs(denom) > 1e-12 and abs(np.dot(y, Hy)) > 1e-12:
            H += (np.outer(s, s) / (denom+ 1e-12)) - (np.outer(Hy, yH) / (np.dot(y, Hy) + 1e-12))
       
        x = x_new
    plot_results(f, initial_point, "dfp", f_vals, grad_norms, x_history)
    return x


def bfgs(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    x = initial_point.copy()
    n = len(x)
    H = np.eye(n)
    tol = 1e-6
    max_iter = 1000
    f_vals = []
    x_history = []
    grad_norms = []
    for _ in range(max_iter):
        f_vals.append(f(x))
        x_history.append(x.copy())
        grad = d_f(x)
        grad_norms.append(np.linalg.norm(grad))
        if np.linalg.norm(grad) <= tol:
            break

        d = -H @ grad

        phi = lambda a: f(x + a * d)
        d_phi = lambda a: np.dot(d_f(x + a * d), d)
        if d_phi(0) >= 0:
            d = -grad
        alpha = bisection_line_search(phi, d_phi)

        s = alpha * d
        x_new = x + s
        y = d_f(x_new) - grad

        rho = 1.0 / (np.dot(y, s) + 1e-12)
        I = np.eye(n)
        A = I - rho * np.outer(s, y)
        B = I - rho * np.outer(y, s)
        H = A @ H @ B + rho * np.outer(s, s)

        x = x_new
    plot_results(f, initial_point, "bfgs", f_vals, grad_norms, x_history)
    return x

def bisection_line_search(
    phi: Callable[[float], float],
    d_phi: Callable[[float], float],
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    c1 = 0.001
    c2 = 0.1
    alpha = 0
    beta = 1e6
    t = 1

    f_0 = phi(0)
    dphi_0 = d_phi(0)

    for _ in range(max_iter):
        f_t = phi(t)
        dphi_t = d_phi(t)

        if f_t > f_0 + c1 * t * dphi_0:
            beta = t
        elif dphi_t < c2 * dphi_0:
            alpha = t
        else:
            return t

        t = (alpha + beta) / 2.0
        if abs(beta - alpha) < tol:
            return t

    return t
def plot_results(f, inital_point, condition, f_vals, grad_norms, x_history):
    os.makedirs("plots", exist_ok=True)
    # Function value plot
    plt.figure()
    plt.plot(f_vals)
    plt.xlabel('Iterations')
    plt.ylabel('Function Value')
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_vals.png")
    plt.close()
    # Gradient norm plot
    plt.figure()
    plt.plot(grad_norms)
    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_grad.png")
    plt.close()

    # contour plot for 2D cases
    if inital_point.ndim == 1 and len(inital_point) == 2:
        x_hist = np.array(x_history)
       
        #setting up the plot boundaries.
        x_min, x_max = x_hist[:, 0].min(), x_hist[:, 0].max() #x_hist[:, 0] Stores all x-coordinates.
        y_min, y_max = x_hist[:, 1].min(), x_hist[:, 1].max() #x_hist[:, 1] #Stores all y-coordinates.
        #Padding=max(10^−5 ,0.5×(max value−min value)) to Improve visualization clarity
        x_pad = max(1e-5, 0.5 * (x_max - x_min))
        y_pad = max(1e-5, 0.5 * (y_max - y_min))
        #Creating final grid of 100x100 points with padding
        #np.linspace(start, stop, num) creates evenly spaced values between start and stop
        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 100)
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = np.array([f(np.array([x, y])) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)
        plt.figure()
        plt.contour(X, Y, Z, levels=50)
       
        plt.plot(x_hist[:, 0], x_hist[:, 1], 'r--o')

        plt.quiver(x_hist[:-1, 0], x_hist[:-1, 1],#Start points
                    x_hist[1:, 0] - x_hist[:-1, 0],#X-direction change
                      x_hist[1:, 1] - x_hist[:-1, 1], # Y-direction change
                   angles='xy', scale_units='xy', scale=1, color='blue', width=0.003,headwidth=8, headlength=10, headaxislength=7)
        
        
        # Highlight the start point
        plt.scatter(x_hist[0, 0], x_hist[0, 1], s=150, 
                   edgecolors='black', facecolors='lime', 
                   marker='o', label='Start Point')
        # Highlight the end point
        plt.scatter(x_hist[-1, 0], x_hist[-1, 1], s=200,
                   edgecolors='black', facecolors='red', 
                   marker='X', label='Final Point')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Contour Plot: {condition}')
        plt.savefig(f"plots/{f.__name__}_{np.array2string(inital_point)}_{condition}_cont.png")
        plt.close()
