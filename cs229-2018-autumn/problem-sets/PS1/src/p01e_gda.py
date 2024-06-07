import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    clf = GDA()
    clf.fit(x_train, y_train)

    # Plot data and decision boundary
    util.plot(x_train, y_train, clf.theta, 'output/p01e_{}.png'.format(pred_path[-5]))


    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    
    # clf.predict()    
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n+1)
        ## use phi, miu0, miu1 and sigma to compute theta
        phi = np.mean(y==1)
        ## x[y==0]: filter row where y==0
        u_0 = np.sum(x[y==0], axis=0)/np.sum(y==0)
        u_1 = np.sum(x[y==1], axis=0)/np.sum(y==1)
        sigma = ((x[y==0]-u_0).T @ (x[y==0]-u_0) + \
                    (x[y==1]-u_1).T @ (x[y==1]-u_1))/m

        ##compute theta
        sigma_inv = np.linalg.inv(sigma)
        self.theta[0] = (u_0+u_1).T @ sigma_inv @ (u_0-u_1)/2 - np.log((1-phi)/phi)
        self.theta[1:] = sigma_inv @ (u_1-u_0)
        return self.theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            ## addistional dimension in x with value 1s to fit theta0
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1/(1 + np.exp(-(x@self.theta)))
        # *** END CODE HERE
