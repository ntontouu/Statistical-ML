# Parts of the code were taken/slightly adjusted from https://github.com/oniani/ai/blob/main/model/ml/gaussian_naive_bayes.py

import numpy as np

from utils.data import data

class GaussianNaiveBayes:
    def fit(self, features: np.ndarray, labels: np.ndarray):
        """Fits the Gaussian Naive Bayes model."""

        self.labels = labels
        self.unique_labels = np.unique(labels)
        self.params = []
        for label in self.unique_labels:
            label_features = features[self.labels == label]
            self.params.append([(col.mean(), col.var()) for col in label_features.T])

    def likelihood(self, data: float, mean: float, var: float):
        """Calculates the Gaussian likelihood of the data with the given mean and variance."""
        var += 1e-9
        coeff = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-((data - mean) ** 2 / (2 * var)))
        return coeff * exponent

    def predict(self, features: np.ndarray):
        """Performs inference using Bayes' Theorem:  P(A | B) = P(B | A) * P(A) / P(B)."""

        num_samples, _ = features.shape
        predictions = np.empty(num_samples)
        for idx, feature in enumerate(features):
            posteriors = []
            for label_idx, label in enumerate(self.unique_labels):
                prior = np.log((self.labels == label).mean())

                # P(a0, a1, a2 | B) = P(a0 | B) * P(a1 | B) * P(a2 | B)
                pairs = zip(feature, self.params[label_idx])
                likelihood = np.sum([np.log(self.likelihood(f, m, v)) for f, (m, v) in pairs])

                posteriors.append(prior + likelihood)

            # Store the label with the largest posterior probability
            predictions[idx] = self.unique_labels[np.argmax(posteriors)]
        return predictions

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    data_obj = data()
    data_obj.load_arff("datasets/contact-lenses.arff")
    # data_obj.load_csv("datasets/iris.csv")
    train_features, test_features, train_labels, test_labels = train_test_split(
        data_obj.features, data_obj.labels, test_size=0.3, random_state=0
    )

    gnb = GaussianNaiveBayes()
    gnb.fit(train_features, train_labels)
    predictions = gnb.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)

    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Mislabeled points: {(predictions != test_labels).sum()}/{test_features.shape[0]}")