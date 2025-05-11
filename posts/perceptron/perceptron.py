import torch
from matplotlib import pyplot as plt


def perceptron_data(n_points = 300, noise = 0.2, p_dims = 2):
    
    y = torch.arange(n_points) >= int(n_points/2)
    X = y[:, None] + torch.normal(0.0, noise, size = (n_points,p_dims))
    X = torch.cat((X, torch.ones((X.shape[0], 1))), 1)

    # convert y from {0, 1} to {-1, 1}
    
    return X, y

def plot_perceptron_data(X, y, ax):
    assert X.shape[1] == 3, "This function only works for data created with p_dims == 2"
    targets = [0, 1]
    markers = ["o" , ","]
    for i in range(2):
        ix = y == targets[i]
        ax.scatter(X[ix,0], X[ix,1], s = 20,  c = 2*y[ix]-1, facecolors = "none", edgecolors = "darkgrey", cmap = "BrBG", vmin = -2, vmax = 2, alpha = 0.5, marker = markers[i])
    ax.set(xlabel = r"$x_1$", ylabel = r"$x_2$")


def draw_line(w, x_min, x_max, ax, **kwargs):
    w_ = w.flatten()
    x = torch.linspace(x_min, x_max, 101)
    y = -(w_[0]*x + w_[2])/w_[1]
    l = ax.plot(x, y, **kwargs)

def train_perceptron(X, y, perceptron_model, optimizer, max_iterations=1000):
    """
    Train the Perceptron model on the given dataset X with labels y. The function performs 
    gradient descent updates for a specified number of iterations (max_iterations) or until the model 
    achieves zero loss. The loss progression is recorded and returned.

    ARGUMENTS:
        X (torch.Tensor): The feature matrix, where X.size() = (n_points, p_dims + 1).
        y (torch.Tensor): The labels for the data points. y.size() = (n_points,).
        perceptron_model (Perceptron): The perceptron model to be trained.
        optimizer (PerceptronOptimizer): The optimizer that will perform weight updates.
        max_iterations (int): The maximum number of iterations to train the model. Default is 1000.

    RETURNS:
        loss_vec (list): A list of loss values recorded after each update where a change was made. This represents 
                         the training progress.
    """
    loss_vec = []
    n = X.shape[0]
    loss = 1
    iteration = 0

    while iteration < max_iterations and loss > 0:
        i = torch.randint(n, size=(1,))
        x_i = X[[i], :]
        y_i = y[i]
        
        local_loss = perceptron_model.loss(x_i, y_i)
        
        if local_loss > 0: 
            optimizer.step(x_i, y_i)
        
        if local_loss > 0:
            loss = perceptron_model.loss(X, y)
            loss_vec.append(loss)
        
        iteration += 1

    return loss_vec

def plot_loss_curve(loss_vec):
    plt.plot(loss_vec, color="slategrey")
    plt.scatter(torch.arange(len(loss_vec)), loss_vec, color="slategrey")
    plt.xlabel("Perceptron Iteration (Updates Only)")
    plt.ylabel("Loss")
    plt.show()

    
class LinearModel:

    def __init__(self):
        self.w = None 

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        # your computation here: compute the vector of scores s
        # basically returning dot product of slef.w, x
        return  X @ self.w 

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        scores = self.score(X)
        y_hat = (scores > 0).float() 
        return y_hat




class Perceptron(LinearModel):

    def loss(self, X, y):
        """
        Compute the misclassification rate. A point i is classified correctly if it holds that s_i*y_i_ > 0, where y_i_ is the *modified label* that has values in {-1, 1} (rather than {0, 1}). 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

            y, torch.Tensor: the target vector.  y.size() = (n,). The possible labels for y are {0, 1}
        
        HINT: In order to use the math formulas in the lecture, you are going to need to construct a modified set of targets and predictions that have entries in {-1, 1} -- otherwise none of the formulas will work right! An easy to to make this conversion is: 
        """
        y_ = 2*y - 1
        misclassified = (self.score(X) * y_ <= 0).float()
        return misclassified.mean().item()
       
    def grad(self, X, y):
        y_ = 2*y - 1
        misclassified = (self.score(X) * y_ <= 0).float()
        return -1* misclassified * y_ * X
    



class PerceptronOptimizer:

    def __init__(self, model):
        self.model = model 
    
    def step(self, X, y):
        """
        Compute one step of the perceptron update using the feature matrix X 
        and target vector y. 
        return 
        0:
        1:
        for i in range(X.shape[0]):
            x_i = X[i]
            y_i = y[i]
            self.model.w -= self.model.grad(x_i, y_i)
        """
        gradients = self.model.grad(X, y)
        grad_sum = gradients.sum(dim=0)
        self.model.w = self.model.w - grad_sum