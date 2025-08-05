from typing import Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Extra, Field#, validate_arguments

#Reference: https://github.com/omadson/fuzzy-c-means


class FCM(BaseModel):
    r"""Fuzzy C-means Model

    Attributes:
        n_clusters (int): The number of clusters to form as well as the number
        of centroids to generate by the fuzzy C-means.
        max_iter (int): Maximum number of iterations of the fuzzy C-means
        algorithm for a single run.
        m (float): Degree of fuzziness: $m \in (1, \infty)$.
        error (float): Relative tolerance with regards to Frobenius norm of
        the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        random_state (Optional[int]): Determines random number generation for
        centroid initialization.
        Use an int to make the randomness deterministic.
        trained (bool): Variable to store whether or not the model has been
        trained.

    Returns:
        FCM: A FCM model.

    Raises:
        ReferenceError: If called without the model being trained
    """

    class Config:
        extra = Extra.allow
        arbitrary_types_allowed = True

    n_clusters: int = Field(5, ge=1, le=100)
    max_iter: int = Field(150, ge=1, le=1000)
    m: float = Field(2.0, ge=1.0)
    error: float = Field(1e-5, ge=1e-9)
    random_state: Optional[int] = None
    trained: bool = Field(False)

    #@validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(self, X: NDArray, X_Weight: NDArray) -> None:
        """Train the fuzzy-c-means model

        Args:
            X (NDArray): Training instances to cluster.
        """
        self.rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        self.u = self.rng.uniform(size=(n_samples, self.n_clusters))
        self.u = self.u / np.tile(
            self.u.sum(axis=1)[np.newaxis].T, self.n_clusters
        )
        for _ in range(self.max_iter):
            u_old = self.u.copy()
            self._centers = FCM._next_centers(X, self.u, self.m)
            self.u = self.soft_predict(X,X_Weight)
            # Stopping rule
            if np.linalg.norm(self.u - u_old) < self.error:
                break
        self.trained = True

    #@validate_arguments(config=dict(arbitrary_types_allowed=True))
    def soft_predict(self, X: NDArray, X_Weight: NDArray) -> NDArray:
        """Soft predict of FCM

        Args:
            X (NDArray): New data to predict.

        Returns:
            NDArray: Fuzzy partition array, returned as an array with
            n_samples rows and n_clusters columns.
        """
        temp = FCM._dist(X, self._centers, X_Weight) ** (2 / (self.m - 1))
        #print(temp.shape)  #(784, 6)
        
        denominator_ = temp.reshape((X.shape[0], 1, -1)).repeat(
            temp.shape[-1], axis=1
        )
        denominator_ = temp[:, :, np.newaxis] / denominator_
        return 1 / denominator_.sum(2)

    #@validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(self, X: NDArray,X_Weight: NDArray) -> NDArray:
        """Predict the closest cluster each sample in X belongs to.

        Args:
            X (NDArray): New data to predict.

        Raises:
            ReferenceError: If it called without the model being trained.

        Returns:
            NDArray: Index of the cluster each sample belongs to.
        """
        if self._is_trained():
            X = np.expand_dims(X, axis=0) if len(X.shape) == 1 else X
            return self.soft_predict(X,X_Weight).argmax(axis=-1)
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    def _is_trained(self) -> bool:
        if self.trained:
            return True
        return False

    @staticmethod
    def _dist(A: NDArray, B: NDArray, A_Weight: NDArray) -> NDArray:
        """Compute the euclidean distance two matrices"""
        #print(((A[:, None, :] - B) ** 2).shape) #(784, 6, 135984)
        return np.sqrt(np.einsum("ijk->ij", (A[:, None, :] - B) ** 2* A_Weight[None,None,:]))

    @staticmethod
    def _next_centers(X: NDArray, u: NDArray, m: float):
        """Update cluster centers"""
        um = u**m
        return (X.T @ um / np.sum(um, axis=0)).T

    @property
    def centers(self) -> NDArray:
        if self._is_trained():
            return self._centers
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    @property
    def partition_coefficient(self) -> float:
        """Partition coefficient

        Equation 12a of
        [this paper](https://doi.org/10.1016/0098-3004(84)90020-7).
        """
        if self._is_trained():
            return np.mean(self.u**2)
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )

    @property
    def partition_entropy_coefficient(self):
        if self._is_trained():
            return -np.mean(self.u * np.log2(self.u))
        raise ReferenceError(
            "You need to train the model. Run `.fit()` method to this."
        )
        
        

if __name__ == '__main__':
    #mporting libraries
    
    #matplotlib inline
    #import numpy as np
    #from fcmeans import FCM
    from matplotlib import pyplot as plt
    
    #creating artificial data set
    
    
    #============Weighted FCM==================
    
    n_samples = 300
    
    X = np.concatenate((
        np.random.normal((-1, -2), size=(n_samples, 2)),
        np.random.normal((2, 2), size=(n_samples, 2))
    ))
    
    #fitting the fuzzy-c-means
    
    w_value=1
    X_Weight=np.zeros(X.shape[1])
    X_Weight[1]=100
    X_Weight[0]=100
    
    fcm = FCM(n_clusters=2)
    fcm.fit(X,X_Weight=X_Weight)
    
    #showing results
    
    # outputs
    fcm_centers = fcm.centers
    fcm_labels = fcm.predict(X,X_Weight)
    
    # plot result
    f, axes = plt.subplots(1, 2, figsize=(11,5))
    axes[0].scatter(X[:,0], X[:,1])#, alpha=.1)
    axes[1].scatter(X[:,0], X[:,1], c=fcm_labels)#, alpha=.1)
    axes[1].scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='w')
    #plt.savefig('images/basic-clustering-output.jpg')
    plt.show()

