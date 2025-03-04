class NeuralNetwork:
    def _init_(self):
        self.inputs = [0.05, 0.10]
        self.targets = [0.1, 0.99]
        self.learning_rate = 0.5
        self.w1, self.w2, self.w3, self.w4 = 0.15, 0.20, 0.25, 0.30
        self.b1, self.b2 = 0.35, 0.35
        self.w5, self.w6, self.w7, self.w8 = 0.40, 0.45, 0.50, 0.55
        self.b3, self.b4 = 0.60, 0.60
    def sigmoid(self, x):
        return 1 / (1 + (2.71828 ** -x))
    def sigmoid_derivative(self, x):
        return x * (1 - x) 
    def forward(self):
        self.h1_input = self.inputs[0] * self.w1 + self.inputs[1] * self.w3 + self.b1
        self.h2_input = self.inputs[0] * self.w2 + self.inputs[1] * self.w4 + self.b2   
        self.h1_output = self.sigmoid(self.h1_input)
        self.h2_output = self.sigmoid(self.h2_input)  
        self.o1_input = self.h1_output * self.w5 + self.h2_output * self.w7 + self.b3
        self.o2_input = self.h1_output * self.w6 + self.h2_output * self.w8 + self.b4 
        self.o1_output = self.sigmoid(self.o1_input)
        self.o2_output = self.sigmoid(self.o2_input)
    def backward(self):
        error_o1 = self.targets[0] - self.o1_output
        error_o2 = self.targets[1] - self.o2_output 
        grad_o1 = error_o1 * self.sigmoid_derivative(self.o1_output)
        grad_o2 = error_o2 * self.sigmoid_derivative(self.o2_output) 
        grad_h1 = (grad_o1 * self.w5 + grad_o2 * self.w6) * self.sigmoid_derivative(self.h1_output)
        grad_h2 = (grad_o1 * self.w7 + grad_o2 * self.w8) * self.sigmoid_derivative(self.h2_output)  
        self.w5 += self.learning_rate * self.h1_output * grad_o1
        self.w6 += self.learning_rate * self.h2_output * grad_o1
        self.w7 += self.learning_rate * self.h1_output * grad_o2
        self.w8 += self.learning_rate * self.h2_output * grad_o2
        self.w1 += self.learning_rate * self.inputs[0] * grad_h1
        self.w2 += self.learning_rate * self.inputs[0] * grad_h2
        self.w3 += self.learning_rate * self.inputs[1] * grad_h1
        self.w4 += self.learning_rate * self.inputs[1] * grad_h2 
        self.b3 += self.learning_rate * grad_o1
        self.b4 += self.learning_rate * grad_o2
        self.b1 += self.learning_rate * grad_h1
        self.b2 += self.learning_rate * grad_h2
    def train(self, epochs=1):
        for _ in range(epochs):
            self.forward()
            self.backward()
        print("Updated Weights:")
        print(f"w1: {self.w1}, w2: {self.w2}, w3: {self.w3}, w4: {self.w4}")
        print(f"w5: {self.w5}, w6: {self.w6}, w7: {self.w7}, w8: {self.w8}")
        print("Final Outputs:", self.o1_output, self.o2_output)
nn = NeuralNetwork()
nn.train()