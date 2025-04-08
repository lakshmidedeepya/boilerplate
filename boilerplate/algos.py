from typing import Callable, Literal
from numpy.typing import NDArray
import numpy as np

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

    for k in range(max_iter):
        if np.linalg.norm(grad) < tol:
            break
        phi = lambda alpha: f(x + alpha * d)
        d_phi = lambda alpha: np.dot(d_f(x + alpha * d), d)

        alpha = bisection_line_search(phi, d_phi, 0, 1)
        x_new = x + alpha * d
        prev_grad = grad.copy()
        grad = d_f(x_new)


        if k >= n - 1:
            x=x_new
            k=0
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

    return x


def sr1(
    initial_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    x = initial_point.copy()
    n = len(x)
    B = np.eye(n)
    tol = 1e-6
    max_iter = 1000

    for _ in range(max_iter):
        grad = d_f(x)
        if np.linalg.norm(grad) < tol:
            break

        d = -np.linalg.solve(B, grad)

        phi = lambda a: f(x + a * d)
        d_phi = lambda a: np.dot(d_f(x + a * d), d)
        if d_phi(0) >= 0:
            d = -grad
        alpha = bisection_line_search(phi, d_phi, 0, 1)

        s = alpha * d
        x_new = x + s
        y = d_f(x_new) - grad

        Bs = B @ s
        y_minus_Bs = y - Bs
        denom = np.dot(y_minus_Bs, s)

        if abs(denom) > tol * np.linalg.norm(s) * np.linalg.norm(y_minus_Bs):
            B += np.outer(y_minus_Bs, y_minus_Bs) / denom

        x = x_new

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
    max_iter = 1000

    for _ in range(max_iter):
        grad = d_f(x)
        if np.linalg.norm(grad) < tol:
            break

        d = -H @ grad

        phi = lambda a: f(x + a * d)
        d_phi = lambda a: np.dot(d_f(x + a * d), d)
        if d_phi(0) >= 0:
            d = -grad
        alpha = bisection_line_search(phi, d_phi, 0, 1)

        s = alpha * d
        x_new = x + s
        y = d_f(x_new) - grad

        Hy = H @ y
        denom = np.dot(y, s)

        if abs(denom) > tol * np.linalg.norm(s) * np.linalg.norm(y):
            H += np.outer(s, s) / denom - np.outer(Hy, Hy) / (np.dot(y, Hy) + 1e-12)

        x = x_new

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

    for _ in range(max_iter):
        grad = d_f(x)
        if np.linalg.norm(grad) < tol:
            break

        d = -H @ grad

        phi = lambda a: f(x + a * d)
        d_phi = lambda a: np.dot(d_f(x + a * d), d)
        if d_phi(0) >= 0:
            d = -grad
        alpha = bisection_line_search(phi, d_phi, 0, 1)

        s = alpha * d
        x_new = x + s
        y = d_f(x_new) - grad

        rho = 1.0 / (np.dot(y, s) + 1e-12)
        I = np.eye(n)
        A = I - rho * np.outer(s, y)
        B = I - rho * np.outer(y, s)
        H = A @ H @ B + rho * np.outer(s, s)

        x = x_new

    return x


def bisection_line_search(
    phi: Callable[[float], float],
    d_phi: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    c1 = 0.001
    c2 = 0.1
    alpha = a
    beta = 1e6
    t = b

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
