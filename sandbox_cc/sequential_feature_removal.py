from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import multiprocessing
print('here')
class SBS():
    """
    from python machine learning book
    Sequentially remove features and monitor change in accuracy
    """
    # SEQUENTIAL FEATURE REMOVAL
    def __init__(self, estimator, k_features,
        scoring=accuracy_score,
        test_size = 0.2,
        test_loops=10,
        random_state = None,
        threads = 10
    ):
        threads = 1
        print('yadda2', 'ignore thread param')
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        self.loops = test_loops
        self.threads = threads

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # dim = number of features
        dim = X_train.shape[1]

        # indices for each dim
        self.indices_ = tuple(range(dim))

        # make the current set of indices for each score (self.scores = y coord, self.subsets_ = x coord)
        self.subsets_ = [self.indices_]

        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)

        # score for all of the data
        self.scores_ = [score]        

        while dim > self.k_features:
            print('current dim: ', dim)
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim-1):
                # slice dim - 1 features from indices
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)  # score for this subset
                subsets.append(p)  # indices used for this subset

            best = np.argmax(scores)  # which resulted in best score
            self.indices_ = subsets[best]  # these are the indices we should use for best results
            self.subsets_.append(self.indices_)

            dim = len(self.indices_)  # same as dim -= 1

            self.scores_.append(scores[best])

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _ind_calc(self, X_train, y_train, X_test, y_test, indices, queue=None):
        # fit data
        self.estimator.fit(X_train[:, indices], y_train)
        # predict data
        y_pred = self.estimator.predict(X_test[:, indices])
        # get accuracy for these settings
        score = self.scoring(y_test, y_pred)
        if queue is not None:
            queue.put(score)
        return score

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        """
        indices => which columns to slice for fitting data
        """
        avg_sc = []

        total_iter = 0
        while total_iter < self.loops:
            njobs = min(self.threads, self.loops - total_iter)
            if njobs > 1:
                queue = multiprocessing.Queue()
                jobs = []
                for j in range(njobs):
                    p = multiprocessing.Process(
                        target=self._ind_calc,
                        args=(X_train, y_train, X_test, y_test, indices,queue,)
                    )
                    p.start()
                    jobs.append(p)
                for j in jobs:
                    j.join()
                while True:
                    if queue.empty() == True:
                        break
                    avg_sc.append(queue.get_nowait())
            else:
                avg_sc.append(self._ind_calc(X_train, y_train, X_test, y_test, indices))

            total_iter += njobs
        return np.array(avg_sc).mean()  # score