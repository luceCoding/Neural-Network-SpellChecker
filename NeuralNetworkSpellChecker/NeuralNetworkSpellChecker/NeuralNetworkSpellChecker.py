import numpy as np

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)

TrainingInput = ['beautyful']
ExpectedResult = ['beautiful']

Input = np.array([])
Output = np.array([])

#convert the inputs and outputs to values between 0 and 1
for i in ExpectedResult:
    count = 0
    tmp = np.array([])
    for j in i:
        count += 1
        tmp = np.append(tmp, ord(j)/127)
    for k in range(0,9-count):
        tmp = np.append(tmp, 0/127)
    Output = np.append(Output, tmp)

for i in TrainingInput:
    count = 0
    tmp = np.array([])
    for j in i:
        count += 1
        tmp = np.append(tmp, ord(j)/127)
    for k in range(0,9-count):
        tmp = np.append(tmp, 0/127)
    Input = np.append(Input,tmp)

print (Output)
print (Input)

X = Input
y = Output

alpha = 0.01
print ("\nTraining With Alpha:" + str(alpha))
np.random.seed(None)

# initialize weights randomly with mean 0
synapse_0 = 2*np.random.random((9,9)) - 1

for j in range(100000):

    # forward propagation
    layer_0 = X
    layer_1 = sigmoid(np.dot(layer_0,synapse_0))

    # how much did we miss?
    layer_1_error = layer_1 - y

    if (j% 10000) == 0:
        print ("Error:" + str(np.mean(np.abs(layer_1_error))))

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)
    #synapse_0_derivative = np.dot(layer_0.T,layer_1_delta)

    # update weights
    synapse_0 -= alpha * (layer_1.T.dot(layer_1_delta))
       
print ("TRAINING COMPLETED")
print ("\nLayer 0")
print (layer_0)
print ("\nLayer 1")
print (layer_1)
print ("\nInput")
print (Input)
print ("\nExpected Output")
print (Output)