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

