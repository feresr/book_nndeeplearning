import NeuralNetwork
import FerNet
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)

nn = FerNet.Fernet([784, 20, 10])
nn.SGD(training_data, 30, 10, 0.5,
 lmbda = 5.0,
 evaluation_data=validation_data,
 monitor_evaluation_accuracy=True,
 monitor_evaluation_cost=True,
 monitor_training_accuracy=True,
 monitor_training_cost=True)

