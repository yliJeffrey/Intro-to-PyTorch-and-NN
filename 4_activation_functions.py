# Part 4
# Activation Functions -- introduce non-linearity

# ReLU function - if a number is negative return 0, otherwise return the number with no changes
print('ReLU(-3) = ' + str(0))
print('ReLU(0) = ' + str(0))
print('ReLU(3) = ' + str(3) + '\n')

# define the ReLU function
def ReLU(x):
    return max(0, x)

ReLU_output = ReLU(-2 + 1 + .5)
print(ReLU_output)