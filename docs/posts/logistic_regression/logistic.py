import torch

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
            self.w = torch.rand(X.size(1))

        # your computation here: compute the vector of scores s
        # basically returning dot product of self.w, x
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
    
class LogisticRegression(LinearModel):
    
    def loss(self, X, y):
        """
        Compute the logistic loss (empirical risk) given by the formula:
        given by:
            L(X, y) = - 1/n * Σ (y_i * log(sigmoid(s_i)) + (1 - y_i) * log(1 - sigmoid(s_i)))

        ARGUMENTS:
            X, torch.Tensor: feature matrix (n, p)
            y, torch.Tensor: labels (n,)

        RETURNS:
            loss, torch.Tensor: scalar loss
        """
        scores = self.score(X)
        sigmoid_scores = torch.sigmoid(scores)
        loss = torch.mean(-y * torch.log(sigmoid_scores) - (1 - y) * torch.log(1 - sigmoid_scores))
        return loss
    
    def grad(self, X, y):
        """
        Compute the gradient of the logistic regression loss with respect to the model weights.
        
        The gradient formula used is:
            grad = 1/n * Σ (X_i * (σ(s_i) - y_i)),
        where σ(s_i) is the sigmoid of the score s_i for the ith data point.

        ARGUMENTS:
            X (torch.Tensor): The feature matrix of size (n, p).
            y (torch.Tensor): The target labels of size (n,).
        
        RETURNS:
            grad (torch.Tensor): The gradient of the loss with respect to the weights.
        """
        n = X.size(0)
        scores = self.score(X)
        sigma_scores = torch.sigmoid(scores)
        error_term = sigma_scores - y
        grad = (X.T @ error_term) / n
        return grad
    
class GradientDescentOptimizer():

    def __init__(self, model):
        """
        Initializes the GradientDescentOptimizer.

        This optimizer keeps track of the model and the previous weights .

        ARGUMENTS:
            model: The model object (i.e., LogisticRegression) to optimize.
        """
        self.model = model
        self.prev_w = None
    
    def step(self, X, y, alpha = 0.1, beta=0.9):
        """
        Perform a single step of gradient descent to update the model weights.

        This method updates the model weights using the computed gradient along with given alpha and beta values.
        
        The weight update formula is:
            w{k+1} = wk - alpha * grad + beta * (wk - w{k-1}),
        where `wk` is the current weights, `w{k-1}` is the previous weights, 
        `grad` is the gradient of the loss.

        ARGUMENTS:
            X (torch.Tensor): The feature matrix of size (n, p).
            y (torch.Tensor): The target labels of size (n,).
            alpha (float, optional): The learning rate. Default is 0.1.
            beta (float, optional): The momentum factor. Default is 0.9.

        RETURNS:
            None: The model's weights are updated in-place.
        """
        grad = self.model.grad(X, y)

        # Initialize previous weights if they don't exist yet
        if self.prev_w is None:
            self.prev_w = self.model.w.clone()

        next_w = self.model.w - alpha * grad + beta * (self.model.w - self.prev_w)

        self.prev_w = self.model.w.clone()
        self.model.w = next_w