# Notes on Quadratic Expressions of Random Variables

One approach to this work relies on quadratic expressions of random variables. Computing the distributions of quadratic expressions is complicated; this note reviews some of the literature.

## Motivation

We seek to find (or approximate) the distribution of $f(X)$, where $f: R^n \rightarrow R$ is an objective function and $X \in R^n$ is a random variable. We assume $X$ is gaussian: $X \sim N_n(\mu, \Sigma)$.

Instead of computing the distribution of $f(X)$, we consider the distribution of $q(x)$, where $q$ is a quadratic approximation of $f$ about some point $x_0$:
$$
q(x) = x^T A x + b^T x + c
$$
Where $A$ and $b$ are the Hessian and Jacobian of $f(x)$ at $x_0$. These can be computed cheaply with automatic differentiation.

The quadratic approximation of the objective is a random variable $Q^*$, which is a quadratic expression of $X$:

$$
Q^* = X^T A X + b^T X + c
$$

 $Q^*$ is a 'generalized chi-squared' random variable [[see wiki](https://en.wikipedia.org/wiki/Generalized_chi-squared_distribution)]. However, there is not a general closed form for the pdf of $Q^*$.

## Literature

### R package 'CompQuadForm'

The R package CompQuadForm implements and discusses several algorithms for approximating the cdf of a quadratic form.

### Davies 1980 AS 155

Davies' paper presents 'Algorithm AS 155' which approximates the cfd of a quadratic form. Davies uses a different expression, 
$$
Q = \sum_j \lambda_j X_j + \sigma X_0
$$
where $X_j$ are noncentral chi-squared rvs and $X_0$ is a standard normal rv. I think this is equivalent to the quadratic expression $Q^*$ 

### Mohsenipour 2012

PhD thesis on 'On the Distribution of Quadratic Expressions in Various Types of Random Vectors'. Chapter 1 provides a good review of the field. Chapter 2 gives approximation algorithms for the quadratic expression we are interested in.