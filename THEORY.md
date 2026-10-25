# Naive Bayes Classifier - Theoretical Background & Analysis

## 🔍 Introduction
The Naive Bayes classifier is a family of algorithms based on the assumption of **statistical independence among features**. Its primary advantages include simplicity, fast execution, and the relatively small amount of data required for training.

---

## 📐 Computational Framework & Bayes' Theorem
Classification is based on calculating the posterior probability $P(C_k \mid \mathbf{x})$ for a given class $C_k$ and feature vector $\mathbf{x} = (x_1, \dots, x_n)$:

$$P(C_k \mid \mathbf{x}) = \frac{P(C_k) P(\mathbf{x} \mid C_k)}{P(\mathbf{x})}$$

Due to the feature independence assumption, the likelihood is expressed as a product:

$$P(\mathbf{x} \mid C_k) = \prod_{i=1}^{n} P(x_i \mid C_k)$$

---

## Hybrid Naive Bayes Implementation
This project implements a **Hybrid Naive Bayes** model, which handles different data types in specific ways:
* **Discrete variables:** Calculated using relative frequencies (probabilities) for each class. * **Continuous variables:** Use of the **Normal Distribution (Gaussian Distribution)**, calculating the mean ($\mu$) and the variance ($\sigma^2$):

$$P(x_i \mid C_k) = \frac{1}{\sqrt{2\pi \sigma_{k,i}^2}} \exp\left( -\frac{(x_i - \mu_{k,i})^2}{2\sigma_{k,i}^2} \right)$$

---

## 🪵 Log-Probabilities
To avoid numerical underflow (the multiplication of many very small numbers close to zero) and to handle missing values ​​or zero probabilities, the equation is converted to a **logarithmic scale**:

$$\log P(C_k \mid \mathbf{x}) \propto \log P(C_k) + \sum_{i=1}^{n} \log P(x_i \mid C_k)$$

---

## 📊 Experimental Results
1. **Weather-Forecast Dataset:** Achieved an accuracy of **80%** (1 error / 5 test samples).
2. **Iris Dataset:** Achieved a high accuracy of **93.3%** (3 errors / 45 test samples).

---
**Authors:**
* Thomas Bakogiannis (Student ID: 2211)
* Maria-Alexandra Dontou (Student ID: 2799)
