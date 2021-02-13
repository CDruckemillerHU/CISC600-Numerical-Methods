import math, cmath

          
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
