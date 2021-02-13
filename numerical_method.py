import math, cmath
from matplotlib import pyplot as plt

          
def eulers_method(old_value, slope, step):
    return old_value + slope*step


def true_error(exact, approx):
    return exact - approx


def true_percent_relative_error(exact, approx):
    return true_error(exact, approx)/exact * 100


def approx_relative_error(approx_error, approx):
    return approx_error/approx if approx_error < approx else approx/approx_error


def approx_relative_percent_error(approx_error, approx):
    return approx_relative_error(approx_error, approx) * 100


def iterative_relative_approx_error(current_approx, previous_approx):
    return abs((current_approx - previous_approx) / current_approx)


def iterative_relative_percent_approx_error(current_approx, previous_approx):
    return iterative_relative_approx_error(current_approx, previous_approx) * 100


def func1(x):
    return .12*x**4 + 5*x**3
    

def taylor_series(func, first_value, second_value, *deriv):
    for item in deriv:
        if type(item) is not type(func):
            raise TypeError()
    terms = []
    terms.append(func(first_value))
    for index, d in enumerate(deriv):
        top = d(first_value)
        bottom = math.factorial(index+1)
        full_value = (top/bottom) * (second_value - first_value)**index
        terms.append(full_value)
    print("Terms:", terms)
    return sum(terms)


def taylor_series_values(values, step):
    results = [values[0]]
    for val, index in enumerate(values[1:]):
        top = val
        bottom = math.factorial(index+1)
        results.append((top/bottom) *step**index)
    return sum(results)


def plot_data(func, domain):
    y_vals = []
    for x in domain:
        y_vals.append(func(x))
        
    plt.plot(domain, y_vals)
    plt.show()
    

def bisection_method(func, lower_bound, upper_bound, previous_value):
    print(lower_bound,upper_bound)
    x_r = (lower_bound+upper_bound)/2
    print(x_r)

    if previous_value:
        approx_error = iterative_relative_percent_approx_error(x_r,previous_value)
        print("Error:", approx_error)
        if approx_error < .5: # Make this a variable?
            return x_r
    approx_root = func(x_r)
    print("How close are we to zero?:", approx_root, x_r)
    if approx_root < 0:
        return bisection_method(func, lower_bound, x_r, x_r)
    else:
        return bisection_method(func, x_r, upper_bound, x_r)


def similar_triangles(func, upper, lower):
    f_upper = func(upper)
    f_lower = func(lower)
    return upper - (f_upper * (lower - upper))/(f_lower - f_upper)


def false_position_method(func, lower_bound, upper_bound, previous_value):
    x_r = similar_triangles(func, upper_bound,lower_bound)
    if previous_value:
        approx_error = iterative_relative_percent_approx_error(x_r,previous_value)
        print("Error:", approx_error)
        if approx_error < .05: # Make this a variable?
            return x_r
    approx_root = func(x_r)
    print("How close are we to zero?:", approx_root, x_r)
    if approx_root < 0:
        return false_position_method(func, lower_bound, x_r, x_r)
    else:
        return false_position_method(func, x_r, upper_bound, x_r)


def bracket_method(func, lower_bound, upper_bound, previous_value, is_bisection):
    x_r = (lower_bound+upper_bound)/2 if is_bisection else similar_triangles(func, upper_bound,lower_bound)
    print(x_r)
    if previous_value:
        approx_error = iterative_relative_percent_approx_error(x_r,previous_value)
        print("Error:", approx_error)
        if approx_error < .05: # Make this a variable?
            return x_r
    approx_root = func(x_r)
    print("How close are we to zero?:", approx_root, x_r)
    if approx_root < 0:
        return bracket_method(func, lower_bound, x_r, x_r, is_bisection)
    else:
        return bracket_method(func, x_r, upper_bound, x_r, is_bisection)