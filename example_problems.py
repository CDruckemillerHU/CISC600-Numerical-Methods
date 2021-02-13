import math
from numerical_method import *

def breakpoint():
    print("\n","*"*50,"\n")


#Example 1.1 & 1.2 Numerical & Analytical Solutions to Falling Parachutist
 
print("Example 1.1 & 1.2: Numerical & Analytical Solutions to Falling Parachutist")

g = 9.81 #Gravity
kg = 68.1
drag = 12.5

times = [n for n in range(0,13,2)]


def slope_function(kilos, drag, velocity):
    return g - (drag/kilos)*velocity


print(eulers_method(0, slope_function(kg, drag, 0), 2))
data = []
data.append(0)
# data.append(eulers_method(None, 0, slope_function(kg, drag, 0), 2))
for t in range(len(times)):
    previous_data = data[t]
    data.append(eulers_method( previous_data, slope_function(kg, drag, previous_data), 2))
    
    
print(times)
print(data)
    

breakpoint()
#Example 3.1 Calculation of Errors
#a
#bridge:
print("Example 3.1: Calculation of Errors")
print("Bridge: ", true_error(10_000, 9_999))
print("Rivet: ", true_error(10, 9))
print("Bridge: ", true_percent_relative_error(true_error(10_000, 9_999), 10_000))
print("Rivet: ", true_percent_relative_error(true_error(10, 9), 10))


breakpoint()


#Example 3.2 Error Estimates for Iterative Methods
#Maclaurin Series Expansion

#Error Criteria
error_threshold = .5 * 10**(2-3)
print("Error Threshold: " , str(error_threshold)+"%")

true_value = pow(math.e, .5)

tpre = true_percent_relative_error(true_value, 1.5)

print(str(tpre) + "%")

approx_error = iterative_relative_percent_approx_error(1.5, 1)
print(str(approx_error) + "%")

# 33.3% > .05% so we continue

cnt = 2 # We are starting after our first series expansions
current_approx = 1.5 #This will get updated in our first pass
previous_approx = 1.5 #This will get updated after our first pass

while approx_error > error_threshold:
    current_approx += (.5**cnt)/math.factorial(cnt)
    approx_error = iterative_relative_approx_error(current_approx, previous_approx)*100
    previous_approx = current_approx
    cnt+=1
    print("Approx Error:", str(approx_error) + "%", "\nApproximate Value: ", current_approx, "\nIterations: ", cnt, "\n" )
    
    
    
breakpoint()

#Example 4.1 Taylor Series Approximation of a Polynomial


def fun1(x):
    return -.1*x**4 - .15*x**3 - .5*x**2 -.25*x +1.2

def fun2(x):
    return -.4*x**3 - .45*x**2 - x - .25

def fun3(x): 
    return -1.2*x**2 - .9*x - 1

def fun4(x):
    return -2.4*x - .9

def fun5(x):
    return -2.4

print(taylor_series(fun1, 0, 1, fun2, fun3, fun4, fun5))

breakpoint()




# plot_data(fun4, [x for x in range(10)])

def drag_coeff(c):
    return 668.06/c * (1-math.exp(-.146843*c)) -40

# bisection_method(drag_coeff, 12,16, None)
# breakpoint()
false_position_method(drag_coeff, 12,16,None)
breakpoint()
bracket_method(drag_coeff, 12, 16, None, False)