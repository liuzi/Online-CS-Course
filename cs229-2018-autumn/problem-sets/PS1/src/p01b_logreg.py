import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    
    # *** START CODE HERE ***
    clf = LogisticRegression(eps=1e-5)
    clf.fit(x_train, y_train)
    util.plot(x_train, y_train, clf.theta, 'output/p01b_{}.png'.format(pred_path[-5]))
    y_pred = clf.predict(x_eval)
    np.savetxt(pred_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)

    Parameters
    ----------

    x: numpy.ndarray
        features, (m,n)
    y: numpy.ndarray
        outputs, (m,)
    """

    def _sigmoid(self, x):
        # print(-x @ self.theta)
        return 1 / ( 1 + np.exp(-(x @ self.theta)))

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)

        while True:
            pre_theta = np.copy(self.theta)
            h_theta_x = self._sigmoid(x)
            J_grad = (x.T @ (h_theta_x - y)) / m
            ## element-wise for y_pred instead of dot product
            J_H = (x.T * h_theta_x * (1 - h_theta_x)) @ x / m
            self.theta -= np.linalg.inv(J_H) @ J_grad
            # print(f'loss is {self._loss(y, y_pred)}')
            if np.linalg.norm(self.theta - pre_theta, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self._sigmoid(x)
        # *** END CODE HERE ***
