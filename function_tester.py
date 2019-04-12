import numpy as np
def dot_product(array,multiplier):
    product = 0
    for i in range(len(array)):
        product+= (array[i] * multiplier)
    return product

dot = dot_product(np.array([1,1,1,1]),2)
print(dot)
