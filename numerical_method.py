import math, cmath

          
def eulers_method(old_value, slope, step):
    return old_value + slope*step


def true_factional_relative_error(error,value):
    return error/value if error < value else value/error


def true_percent_relative_error(error,value):
    return true_factional_relative_error(error, value) * 100


def true_error(exact, approximation):
    return exact - approximation

