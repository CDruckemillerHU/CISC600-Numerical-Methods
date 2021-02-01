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
    