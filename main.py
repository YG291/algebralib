"""
Small library that applies linear algebra & calculus concepts computationally.
Author: Yueheng Guan
"""

from numpy import double, array, zeros, exp
from numpy.typing import NDArray
from typing import Callable


def g(x: float) -> float:
    return x**2


def o1_derivative(f: Callable, point: float) -> double:
    """
    Central difference form: mean of forward and backward slopes
    f'(x) = lim_{h->0}[f(x+h)-f(x)]/h = lim_{h->0}[f(x)-f(x-h)]/h
    2f'(x) = lim_{h->0}[f(x+h)-f(x)+f(x)-f(x-h)]/h
    2f'(x) = lim_{h->0}[f(x+h)-f(x-h)]/h
    f'(x) = lim_{h->0}[f(x+h)-f(x-h)]/2h
    """
    point = double(point)
    h = double(10**(-7))  # avoid float precision errors
    return (f(point+h)-f(point-h))/(2*h)


def o2_derivative(f: Callable, point: float) -> double:
    """
    Central difference form: slope of forward and backward slopes
    DOES NOT NEST the o1_derivative to avoid f(x+-2) which may add to precision error
    f''(x) = lim_{h->0}[f'(x+h)-f'(x)]/h
    f''(x) = lim_{h->0}[[f(x+h)-f(x)/h] - [f(x)-f(x-h)]/h]/h
    f''(x) = lim_{h->0}[f(x+h)-2f(x)+f(x-h)]/(h**2)
    """
    point = double(point)
    h = double(10**(-4))  # requires more room due to h**2
    return (f(point+h) - 2*f(point) + f(point-h))/(h**2)


def gaussian(v: NDArray) -> double:
    total = double(0.0)
    for index in range(len(v)):
        total += (v[index])**2
    total = -1 * total
    return exp(total)


def o1_partial_derivative(f: Callable, v: NDArray) -> NDArray:
    """Scalar valued function, R^n domain.
    Central difference form:
    ∂f/∂x_i = lim_{h->0}(f(x+ei*h)-f(x))/h = lim_{h->0}(f(x)-f(x-ei*h))/h
    ∂f/∂x_i = lim_{h->0}[f(x+ei*h)-f(x-ei*h)]/2h

    Precondition:
    - assume that v is in the domain of f, of course
    """
    v_copy = v.copy().astype(double)
    h = double(10 ** (-7))
    grad = zeros(len(v), dtype=double)
    for index in range(len(v)):
        original = v_copy[index]
        v_copy[index] = original+h
        forward_val = f(v_copy)
        v_copy[index] = original-h
        back_val = f(v_copy)
        v_copy[index] = original
        centre = (forward_val - back_val)/(2*h)
        grad[index] = centre
    return grad


def o2_partial_derivative(f: Callable, v: NDArray) -> NDArray:
    """
    Scalar valued function, R^n domain.
    Central difference form:
    (∂^2)f/∂x_j∂x_i = lim_{k->0}[∂f/∂x_i(x+ke_j) - ∂f/∂x_i]/k
    = lim_{k->0}[∂f/∂x_i - ∂f/∂x_i]/k
    Then, sum the values:
    2(∂^2)f/∂x_j∂x_i = lim_{k->0}[∂f/∂x_i(x+ke_j) - ∂f/∂x_i]/k + [∂f/∂x_i - ∂f/∂x_i]/k
    ∂f/∂x_i(x+ke_j) = lim_{k->0}[f(x+eih+ejk)-f(x-eih+ejk)]/2h
    ∂f/∂x_i(x-ke_j) = lim_{k->0}[f(x+eih-ejk)-f(x-eih-ejk)]/2h
    (∂^2)f/∂x_j∂x_i = lim_{k->0}[f(x+eih+ejk)-f(x-eih+ejk)-f(x+eih-ejk)+f(x-eih-ejk)]/4hk
    """
    v_copy = v.copy().astype(double)
    h = double(10 ** (-4))
    k = double(10 ** (-4))  # temporarily hardcoded
    hessian = zeros([len(v), len(v)], dtype=double)
    for index_1 in range(len(v)):
        ivalue = v_copy[index_1]
        for index_2 in range(len(v)):
            jvalue = v_copy[index_2]
            v_copy[index_1] = ivalue+h
            v_copy[index_2] = jvalue+h
            val1 = f(v_copy)
            v_copy[index_1] = ivalue-h
            val2 = f(v_copy)
            v_copy[index_1] = ivalue+h
            v_copy[index_2] = jvalue-h
            val3 = f(v_copy)
            v_copy[index_1] = ivalue-h
            val4 = f(v_copy)
            hessian[index_1][index_2] = (val1 - val2 - val3 + val4)/(4*k*h)
    return hessian

# TODO: Implement adaptive scaling for h and k


if __name__ == '__main__':
    print(o1_partial_derivative(gaussian, array([0.1, 0.2, 0.1], dtype=double)))
    print(o2_partial_derivative(gaussian, array([0.1, 0.2, 0.1], dtype=double)))
