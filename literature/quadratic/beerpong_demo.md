# Beer-pong Demo

This is a demonstration of the quadratic approximation of the pdf of a function of random variables. To generate the figures, run `beerpong_demo.py`.

## Problem setup

In this example, the function is the range of a projectile:
$$
r = v_{init}^2 \sin(2 \theta) / g
$$
where $v_{init}$ is the initial velocity, $\theta$ is the launch angle, and $g$ is the acceleration due to gravity. Assume $v_{init}, \theta$ are jointly normally distributed.

Imagine we are playing a game where we are trying to shoot the ball into a cup. We want to know how likely the ball is to land in the cup. Thus, we care not only about $r$, but about the distribution of $r$.

## The range function, and its quadratic approximation

The figure "Range vs. launch angle" shows the true function $r(\theta)$ (for fixed  $v_{init}$), and its quadratic approximation at a few points.

The figure "Partial derivatives of range" shows the partial derivatives of the range function.

## Range pdf

Examine the figure "Range pdf for different launch angles". These pdfs come from quadratic approximations of the range function.

Specifically, the first four cumulants of the quadratic approximation are calculated - there is a closed-form solution for the cumulants of a quadratic expression of a normal random vector. Then, a Normal-Inverse Gaussian (NIG) distribution is fit to those four cumulants. The NIG distributions are the approximation shown in the figure.

The approximation is good if the standard deviation of the input distribution is small, compared to the 'length' over which the curvature of the function changes. In figure "Range vs. launch angle", we see that the quadratic approximation is close to the true range function for a few degrees. For larger variations in $\theta$, the curvature changes significantly.

The two figures "Range pdf: quad. approx. vs. samples from true distribution" compare the quadratic approximation to a histogram of samples from the true distribution. For stddev($\theta$) = 1 deg, the quadratic approximation is good. The change in curvature of the range function over 1 degree is small. For stddev($\theta$) = 5 deg, the quadratic approximation is worse. The change in curvature of the range function over 5 degrees is larger.

## Range std deviation vs. launch angle

Here is an example of how information about the distribution can be useful in making decisions.

If we want to get the ball into a cup accurately, we want to choose a trajectory (i.e. $v_{init}, \theta$) so that the range distribution is narrow (i.e. has a small standard deviation). Figure "Range standard deviation vs. launch angle, evaluated by quad. approx." shows a Range standard deviation vs. launch angle for two different cases of stddev($v_{init}$).

If stddev($v_{init}$) is large, initial velocity is the dominant source of uncertainty. The lowest stddev($r$) is achieved at $\theta$ = 0 or 90 degrees, where $dr /dv = 0$. (recall figure "Partial derivatives of range").

 If stddev($v_{init}$) is small, launch angle is the dominant source of uncertainty. The lowest stddev($r$) is achieved at $\theta = 45$ degrees, where $dr /d\theta = 0$.