from numerical_method import eulers_method

g = 9.81 #Gravity
kg = 68.1
drag = 12.5

'''

'''
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
    
    