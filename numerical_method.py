import math, cmath
from matplotlib import pyplot as plt
import numpy as np

          
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
 
 
def similar_triangles(func, lower_bound, upper_bound):
    return upper_bound - (func(upper_bound) * (lower_bound - upper_bound))/(func(lower_bound) - func(upper_bound))
    
      
def bracket_method(func, lower_bound, upper_bound, previous_value, error_threshold, is_bisection):
    x_r = (lower_bound + upper_bound) / 2 if is_bisection else similar_triangles(func, lower_bound, upper_bound) # Bisection vs. False Position?
    approx_root = func(x_r)
    print("Approximate Root:", x_r)
    if previous_value:
        approx_error = iterative_relative_percent_approx_error(x_r, previous_value)
        print("Approximate Error:", approx_error)
        if approx_error < error_threshold:
            return x_r
    if approx_root < 0:
        return bracket_method(func, lower_bound, x_r, x_r, error_threshold, is_bisection)
    else:
        return bracket_method(func, x_r, upper_bound, x_r, error_threshold, is_bisection)
    
    
def fixed_point_iteration(initial_value, func, iterations):
    for _ in range(iterations):
        new_val = func(initial_value)
        print(new_val)
        initial_value = new_val
    
    return new_val
        
        
def newton_raphson(initial_value, func, func_prime, iterations):
    for _ in range(iterations):
        new_value = initial_value - ((func(initial_value))/(func_prime(initial_value)))
        print(new_value)
        initial_value = new_value
    return new_value
    

def secant_method(current_value, previous_value, func, iterations):    
    for _ in range(iterations):
        f_x = func(current_value)
        f_prev = func(previous_value)
        print("f(x):", f_x, "f(x-1):", f_prev, "x:", current_value, "x-1:", previous_value)

        new_value = current_value - ((f_x * (previous_value - current_value))/(f_prev - f_x))
        print(new_value)
        previous_value = current_value
        current_value = new_value
        
        
    return new_value
    
    
def modified_newton_raphson(initial_value, func, func_prime, func_double_prime, iterations):
    for _ in range(iterations): 
        f_x = func(initial_value)
        f_prime_x = func_prime(initial_value)
        f_double_prime_x = func_double_prime(initial_value)
        new_value = initial_value - ((f_x * f_prime_x)/(f_prime_x**2 - f_x*f_double_prime_x))
        print(initial_value)
        initial_value = new_value
    
    return new_value


def muller_method(x0, x1, x2, func, previous_approx, error_threshold):
    fx_0 = func(x0)
    fx_1 = func(x1)
    fx_2 = func(x2)
    h0 = x1 - x0
    h1 = x2 - x1
    
    d0 = ((fx_1 - fx_0)/(x1 - x0)) 
    d1 = ((fx_2 - fx_1)/(x2 - x1))
    
    a = ((d1-d0)/(h1+h0))
    b = a * h1 + d1
    c = fx_2
    discrim = math.sqrt(b**2 - 4 * a * c)
    if abs(b + discrim) > abs(b - discrim):
        x3 = x2 + ((-2 * c)/(b + discrim))
    else:
        x3 = x2 + ((-2 * c)/(b - discrim))
    
    error =approx_relative_percent_error(x3 - x2,x3)
    print("Error:", error, "Result:", x3)
    if abs(error) > error_threshold:
        return muller_method(x1, x2, x3, func, x3, error_threshold)
    else:
        return x3


def bairstow_method(r, s, a, func, error_threshold):
    b = [0 for _ in a]
    c = [0 for _ in a]
    b[0] = a[0]
    b[1] = a[1] + b[0] * r
    for i in range(2, len(a)):
        b[i] = a[i] + r*b[i-1] + s*b[i-2]
    
    c[0] = b[0]
    c[1] = b[1] + r*c[0]
    for i in range(2, len(a)-1):
        c[i] = b[i] + r * c[i-1] + s * c[i-2]
    c.pop()
    
    
    c_array = np.array([[c[-2], c[-3]],[c[-1], c[-2]]])
    b_array = np.array([b[-2] * -1, b[-1] * -1])
    detlas = np.linalg.solve(c_array,b_array)
    r_delta = detlas[0]
    s_delta = detlas[1]
    
    r_revised = r + r_delta
    s_revised = s + s_delta
    
    r_error = r_delta/r_revised * 100
    print(r_error, "%")
    s_error = s_delta/s_revised *100
    print(s_error,"%")
    if r_error > error_threshold or s_error > error_threshold:
        bairstow_method(r_revised, s_revised, a, func, error_threshold)
    else:
        print(r_revised, s_revised)
        x_pos = ((r_revised + math.sqrt(r_revised**2 + 4 * s_revised))/(2))
        print(x_pos)
        x_neg = ((r_revised - math.sqrt(r_revised**2 + 4 * s_revised))/(2))
        print(x_neg)
        print(((s_revised)/(r_revised))*-1)
        
    
'''
[3, -.1, -.2, 7.85], 
[.1, 7, -.3, -19.3], 
[.3, -.2, 10, 71.4]
'''    

    
def naive_gauss(equation_matrix): 
    aik = None
    akk = None
    for row_index, rows  in enumerate(equation_matrix):
        for col_index, item in enumerate(rows):
            if row_index == 0 and col_index == 0:
                akk = item
            if row_index == 1 and col_index == 0:
                aik = item
    factor = aik/akk
    print(aik, akk)
    ffe = equation_matrix[0] * factor
    ffe_result = ffe - equation_matrix[1]
    ffe_result = np.array(list(filter(lambda x: x!= 0, ffe_result)))
    print(ffe_result)
    for row_index, rows  in enumerate(equation_matrix):
        for col_index, item in enumerate(rows):
            if row_index == 0 and col_index == 0:
                akk = item
            if row_index == 2 and col_index == 0:
                aik = item
                
def gauss_jordan(equation_matrix):
    for items in equation_matrix[0]:
        equation_matrix[0] = equation_matrix[0]/equation_matrix[0][0]
    
    factor = equation_matrix[0][0]/equation_matrix[1][0]
    elim_factor = equation_matrix[0][0]/factor
    elim_equation = elim_factor*equation_matrix[0]
    equation_matrix[1] = elim_equation - equation_matrix[1]
    
    factor = equation_matrix[0][0]/equation_matrix[2][0]
    elim_factor = equation_matrix[0][0]/factor
    elim_equation = elim_factor*equation_matrix[0]
    equation_matrix[2] = elim_equation - equation_matrix[2]

    for items in equation_matrix[1]:
        equation_matrix[1] = equation_matrix[1]/equation_matrix[1][1]
    
    factor = equation_matrix[1][1]/equation_matrix[0][1]
    elim_factor = equation_matrix[1][1]/factor
    elim_equation = elim_factor*equation_matrix[1]
    equation_matrix[0] = elim_equation - equation_matrix[0]
    
    factor = equation_matrix[1][1]/equation_matrix[2][1]
    elim_factor = equation_matrix[1][1]/factor
    elim_equation = elim_factor*equation_matrix[1]
    equation_matrix[2] = elim_equation - equation_matrix[2]
    
    for items in equation_matrix[2]:
        equation_matrix[2] = equation_matrix[2]/equation_matrix[2][2]
        
    factor = equation_matrix[2][2]/equation_matrix[0][2]
    elim_factor = equation_matrix[2][2]/factor
    elim_equation = elim_factor*equation_matrix[2]
    equation_matrix[0] = elim_equation - equation_matrix[0]
    
    factor = equation_matrix[2][2]/equation_matrix[1][2]
    elim_factor = equation_matrix[2][2]/factor
    elim_equation = elim_factor*equation_matrix[2]
    equation_matrix[1] = elim_equation - equation_matrix[1]

    print(equation_matrix)