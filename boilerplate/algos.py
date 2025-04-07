from typing import Callable, Literal

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np




def conjugate_descent(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    approach: Literal["Hestenes-Stiefel", "Polak-Ribiere", "Fletcher-Reeves"],
) -> NDArray[np.float64]:
        """
        Conjugate Gradient Descent implementation with three different approaches.
        
        Parameters:
            initial_point: Starting point for optimization
            f: Objective function to minimize
            d_f: Gradient of the objective function
            approach: One of "Hestenes-Stiefel", "Polak-Ribiere", or "Fletcher-Reeves"
        
        Returns:
            Optimized point
        """
        x = inital_point.copy()
        grad = d_f(x)
        d = -grad
        prev_grad = grad.copy()
        tol = 1e-6
        max_iter = 1000
        
        for _ in range(max_iter):
            # Line search using bisection method
            phi = lambda alpha: f(x + alpha * d)
            d_phi = lambda alpha: np.dot(d_f(x + alpha * d), d)
            
            # Find optimal step size
            if d_phi(0) >= 0:
                d = -grad  # Reset direction if not a descent
                phi = lambda alpha: f(x + alpha * d)
                d_phi = lambda alpha: np.dot(d_f(x + alpha * d), d)

            alpha = bisection_line_search(phi, d_phi, 0, 1)#3rd arguemnt alpha is 0 and 4th argument b is t
            
            # Update position
            x_new = x + alpha * d
            prev_grad = grad.copy()
            grad = d_f(x_new)
            
            # Check convergence
            if np.linalg.norm(grad) < tol:
                return x_new
                
            # Compute beta according to selected approach
            if approach == "Fletcher-Reeves":
                beta = np.dot(grad, grad) / np.dot(prev_grad, prev_grad)
            elif approach == "Polak-Ribiere":
                beta = np.dot(grad, grad - prev_grad) / np.dot(prev_grad, prev_grad)
            elif approach == "Hestenes-Stiefel":
                beta = np.dot(grad, grad - prev_grad) / np.dot(d, grad - prev_grad)
            
            # Update direction
            d = -grad + beta * d
            
            # Reset to steepest descent if not a descent direction
            if np.dot(d, grad) >= 0:
                d = -grad
                
            x = x_new
            
        return x


def sr1(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
        """
        Symmetric Rank-One (SR1) Quasi-Newton method.
        
        Parameters:
            initial_point: Starting point for optimization
            f: Objective function to minimize
            d_f: Gradient of the objective function
        
        Returns:
            Optimized point
        """
        x = inital_point.copy()
        n = len(x)
        B = np.eye(n)  # Initial Hessian approximation
        tol = 1e-6
        max_iter = 1000
        
        for _ in range(max_iter):
            grad = d_f(x)
            if np.linalg.norm(grad) < tol:
                break
                
            d = -np.linalg.solve(B, grad)
            
            # Line search
            alpha = 0.1
            def phi(a): return f(x + a * d)
            def d_phi(a): return np.dot(d_f(x + a * d), d)
            if d_phi(0) < 0:
                alpha = bisection_line_search(phi, d_phi, 0, 1)
            
            s = alpha * d
            x_new = x + s
            y = d_f(x_new) - grad
            
            # SR1 update
            Bs = B @ s
            y_minus_Bs = y - Bs
            denom = np.dot(y_minus_Bs, s)
            
            if abs(denom) > tol * np.linalg.norm(s) * np.linalg.norm(y_minus_Bs):
                B += np.outer(y_minus_Bs, y_minus_Bs) / denom
                
            x = x_new
            
        return x



def dfp(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Davidon-Fletcher-Powell (DFP) Quasi-Newton method.
    
    Parameters:
        initial_point: Starting point for optimization
        f: Objective function to minimize
        d_f: Gradient of the objective function
    
    Returns:
        Optimized point
    """
    x = inital_point.copy()
    n = len(x)
    H = np.eye(n)  # Initial inverse Hessian approximation
    tol = 1e-6
    max_iter = 1000
    
    for _ in range(max_iter):
        grad = d_f(x)
        if np.linalg.norm(grad) < tol:
            break
            
        d = -H @ grad
        
        # Line search
        alpha = 0.1
        def phi(a): return f(x + a * d)
        def d_phi(a): return np.dot(d_f(x + a * d), d)
        if d_phi(0) < 0:
            alpha = bisection_line_search(phi, d_phi, 0, 1)
        
        s = alpha * d
        x_new = x + s
        y = d_f(x_new) - grad
        
        # DFP update
        Hy = H @ y
        denom = np.dot(y, s)
        
        if abs(denom) > tol * np.linalg.norm(s) * np.linalg.norm(y):
            H += np.outer(s, s) / denom - np.outer(Hy, Hy) / np.dot(y, Hy)
            
        x = x_new
        
    return x


def bfgs(
    inital_point: NDArray[np.float64],
    f: Callable[[NDArray[np.float64]], np.float64 | float],
    d_f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) Quasi-Newton method.
    
    Parameters:
        initial_point: Starting point for optimization
        f: Objective function to minimize
        d_f: Gradient of the objective function
    
    Returns:
        Optimized point
    """
    x = inital_point.copy()
    n = len(x)
    H = np.eye(n)  # Initial inverse Hessian approximation
    tol = 1e-6
    max_iter = 1000
    
    for _ in range(max_iter):
        grad = d_f(x)
        if np.linalg.norm(grad) < tol:
            break
            
        d = -H @ grad
        
        # Line search
        alpha = 0.1
        def phi(a): return f(x + a * d)
        def d_phi(a): return np.dot(d_f(x + a * d), d)
        if d_phi(0) < 0:
            alpha = bisection_line_search(phi, d_phi, 0, 1)
        
        s = alpha * d
        x_new = x + s
        y = d_f(x_new) - grad
        
        # BFGS update using Sherman-Morrison formula
        rho = 1.0 / np.dot(y, s)
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
    """
    Bisection method for inexact line search based on strong Wolfe conditions.
    
    Implements Algorithm 1 from Assignment A2.
    """
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

        # Armijo condition (sufficient decrease)
        if f_t > f_0 + c1 * t * dphi_0:
            beta = t
        # Curvature condition (sufficient progress)
        elif dphi_t < c2 * dphi_0:
            alpha = t
        else:
            return t  # Both conditions satisfied

        t = (alpha + beta) / 2.0

        # Optional: Stop if interval is small enough
        if abs(beta - alpha) < tol:
            return t

    return t  # Return best estimate if max_iter exceeded

          
        
            