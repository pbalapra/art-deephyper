import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from deephyper.problem import HpProblem
import torch


# Define the neural network model (same as before)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=1)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=5, stride=1)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 10)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

# Define the objective function
def objective(params):
    
    eps = params["eps"]
    batch_size = params["batch_size"]
    norm = float(params["norm"])

    # Create the model
    model = Net()
    model.load_state_dict(torch.load('trained_model.pth'))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32)
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32)

    # Create the ART classifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
    )

    # Generate adversarial test examples
    attack = FastGradientMethod(estimator=classifier, eps=eps, batch_size=batch_size, norm=norm)
    x_test_adv = attack.generate(x=x_test)

    # Evaluate the ART classifier on adversarial test examples
    predictions = classifier.predict(x_test_adv)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    #print(accuracy)
    return -accuracy  

if __name__ == "__main__":
    from deephyper.problem import HpProblem
    from deephyper.search.hps import CBO
    from deephyper.evaluator import Evaluator

    # define the variable you want to optimize
    # Step 1: Define the search space for eps
    problem = HpProblem()
    problem.add_hyperparameter((0.01, 0.5, "log-uniform"), "eps")
    problem.add_hyperparameter([2,4,8,16,32,64], "batch_size")
    problem.add_hyperparameter(['inf', '1', '2'], "norm")
   
    # define the evaluator to distribute the computation
    evaluator = Evaluator.create(
        objective,
        method="process",
        method_kwargs={
            "num_workers": 2,
        },
    )

    # define your search and execute it
    search = CBO(problem, evaluator, random_state=42)

    results = search.search(max_evals=100)
    print(results)
