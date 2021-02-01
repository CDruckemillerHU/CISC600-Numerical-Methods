from numerical_method import eulers_method, true_error, true_percent_relative_error
#Example 1.1 & 1.2

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
    

print("\n","*"*50,"\n")
#Example 3.1
#a
#bridge:
print("Example 3.1: Calculation of Errors")
print("Bridge: ", true_error(10_000, 9_999))
print("Rivet: ", true_error(10, 9))
print("Bridge: ", true_percent_relative_error(true_error(10_000, 9_999), 10_000))
print("Rivet: ", true_percent_relative_error(true_error(10, 9), 10))

