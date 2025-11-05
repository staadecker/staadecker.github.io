---
layout: base.liquid
title: "What I learnt in MIT's 14.380 \"Statistics for Economics\" course"
---

# {{ title }}

I think it's a shame how ineffective our post-secondary education system is at ensuring the long-term retention of knowledge. What I learn in class tends to be erased the moment I leave the final exam room. But not today!

In a naive effort to remember what I learnt in my most recent half-semester course at MIT ("14.380: Statistics for Economics"), and because I found the course content quite interesting, I have typed up a version of my class notes that I think should be accessible to a general audience (and my future forgetful self).

I'm also excited to demo some fancy [:expandable boxes](#nutshells) that I think will make the content easier to navigate. Finally, a big shout out goes to Prof. Anna Mikusheva. Her teaching was excellent and her entire course is freely [available](https://ocw.mit.edu/courses/14-381-statistical-method-in-economics-fall-2018/pages/syllabus/) on MIT OpenCourseWare.

### :x Nutshells

Clicking on an expandable box like you just did will reveal additional details that I didn't want to include in the main flow of the text. I'm hoping you like this feature because I spent lots of time getting it to work! Thanks to the fantastic creator Nicky Case for inventing these (see the [nutshell project](https://ncase.me/nutshell/))!

## Course overview
This course covers the mathematical foundations in statistics needed for doing PhD-level economics and econometrics. It is organized as follows:
1. [Part 1](#part-1-probability-distributions-random-variables-and-asymptotic-properties) covers the foundations of statistics, namely random variables and tools to analyze their behaviors as $n \to \infty$, including the delta method, the central limit theorem and O-notation. These tools will be useful to estimate our statistics for large sample sizes.
2. [Part 2](#part-2-statistics) covers statistics and estimators, particularly what makes a "good" estimator and how to estimate "goodness" through tools like the Rao-Cramer bound and Fisher information.
3. [Part 3](#part-3-statistical-tests) covers statistical tests.

Familiarity with calculus, matrix algebra and other math concepts like the [:taylor series expansion](#x-taylor-series) is important.

#### :x Taylor Series

We will often approximate $f(x) - f(a)$ using the [Taylor Series](https://en.wikipedia.org/wiki/Taylor_series):

$$f(x)-f(a)=\frac{f'(a)}{1!}(x-a)+\frac{f''(a)}{2!}(x-a)^2+\ldots$$

When a finite number of terms are used, the approximation is most accurate near $a$.

## Part 1: Probability distributions, random variables, and asymptotic properties

What is probability?

Formally, **probability** is a field of mathematics built upon [:a few axioms](#x-probability-space-and-the-formal-definition-of-probability). The interpretation of these axioms and their meaning in the real-world is a contested topic of philosophy (see [What is The Chance of An Earthquake](https://statistics.berkeley.edu/sites/default/files/tech-reports/611.pdf)), but for our purposes, probability begins with the concept of a random variable.

#### :x Probability Space and the Formal Definition of Probability
The formal definition of probability is based on measure theory. Measure theory helps tie seemingly different mathematical concepts together including magnitude, mass, probability, and geometrical measures like length, area, volume. TODO

The philosophy of frequentist statistics is that there exists an ideal world with potential outcomes $\Omega$. Every experiment produces a realization $\omega \in \Omega$. Here $w$ need not be a number; $\omega$ could be the description "the first die rolled a 5 and the second rolled a 4". Random variables are functions that map these realizations $w$ to the space of real numbers $\mathbb{R}$. Importantly, a random variable also has an associated cumulative probability distribution $F_X(t): \mathbb{R} \to [0,1]$. This cdf requires a probability measure $Pr$, a function that maps sets to the probability of observing that set. While it might be tempting to say $Pr: \Omega \to [0,1]$, it is technically more accurate to define an intermediate set $\mathcal{F}$ which constitutes all possible measurable subsets of $w$ as this richer space $F$ allows for a more interesting $Pr: \mathcal{F} \to [0,1]$. Summarizing:
- $\Omega$ is the space of all potential experimental outcomes
- $\mathcal{F}$ is the set of all measurable subsets of $\Omega$
- $Pr: \mathcal{F} \to [0,1]$ is a probability measure
- $X: \Omega \to \mathbb{R}$ is a random variable mapping outcomes to a number associated with a cdf defined using $Pr$ and $\mathcal{F}$

### Random variables

A **random variable** is a mathematical object *dependent on random events*. Examples include:
- A random variable representing the result of a coin toss (range: $\{H, T\}$).
- A random variable representing the average height of the next 10 people you see.
- A random variable representing the temperature tomorrow.

Importantly, random variables have a probability distribution. $X \sim F_x$ indicates that $X$ is a random variable with cumulative probability distribution (cdf) $F_x(t) = P \{X \leq t\}$. Key tools to work with random variables include the [:probability density function](#x-def-pdf) (pdf) $f_X(t)$, the [:expected value function](#x-def-expected-value) $E[X]$, [:variance](#x-variance) $V(X)$, [:moments](#x-moments), and [:important properties](#x-properties-of-random-variables) of these abstractions.

#### :x Def. pdf

The probability density function is a proxy for the relative likelihood of obtaining a given outcome.
$$f_X(t)=\frac{d}{dt}F_X(t)$$
$$\int_{-\infty}^{\infty} f_X(t) = 1$$

#### :x Def. expected value

The expected value function can be thought of as the _mean_ of an expression $g(X)$ containing random variables.

$$E[g(X)] = \int_{-\infty}^{\infty}g(x)f_X(t)dt$$

In fact, the mean $\mu$ of a random variable is _defined_ as $\mu=E[X]$.

#### :x Variance

Variance represents the _spread_ of the probability density function; how much variance there is across different draws from the same distribution. It is defined as the expected distance squared from the mean:
$$Var(X) = E[(X-E[X])^2]$$

#### :x Moments

The mean and variance are special cases of moments: mathematical objects that help characterize a distribution.

- 1st moment (mean) $E[X]$
- k-th moment $E[X^k]$
- 2nd central moment (variance) $E[(X-E[X])^2]$
- k-th central moment $E\left[(X-E[X])^k\right]$

#### :x Properties of random variables

- Expected value function is linear
	- $E[X+Y] = E[X] + E[Y]$
	- $E[aX] = aE[X]$
- Variance is not
	- $Var(X+Y) \neq Var(X) + Var(Y)$
	- $Var(X+a)=Var(X)$ (shift agnostic)
	- $Var(aX)=a^2Var(X)$
An often used property is:
$$Var(X) =E[X^2] - E[X]^2$$

Note that when $X$ has mean 0, $Var(X) = E[X^2]$.

### Joint distributions
Two random variables could be correlated. For example, $X_t$ and $X_{t+1}$ in a sequence of random events. Or $X$ and $Y$ being characteristics of a person being drawn from a population (e.g. income and education). Or the daily returns of stock $A$ and $B$. Key definitions including the [:joint cumulative distribution](#x-joint-cumulative-distribution), [:marginal distribution](#x-marginal-distribution), [:conditional distributions](#x-conditional-distribution), [:law of iterated expectations](#x-lie), [:covariance](#x-covariance), and [:properties of independent variables](#x-independent-properties).

#### :x Joint cumulative distribution

$$F_{X,Y} (x,y)=P\{ X\leq x,Y\leq y\}$$

#### :x Marginal distribution

$$f_X(x) = P\{X \leq x\} = \int_{-\infty}^{\infty} f(x,y) dy$$

#### :x Conditional distribution

The conditional distribution of $Y$ given $X=x$ is
$$f_{Y|X}(y|x) = \frac{f_{X,Y}(x,y)}{f_X(x)}$$

#### :x LIE

The law of iterated expectations is:

$$E[Y] = E[E[Y|X]]$$

Or more generally:

$$E[g(X)Y] = E\left[g(X) E[Y | X]\right]$$

#### :x Covariance

The covariance operator is symmetric, linear in scaling and shift agnostic. It is defined as

$$\text{cov}(X,Y) = E[(X-E[X])(Y-E[Y])]$$

Two useful properties are:

$$Var(X+Y)=Var(X)+Var(Y)+2\text{cov}(X,Y)$$

and

$$\text{cov}(X,Y) = E[XY] - E[X]E[Y]$$

Notice that if $X$ or $Y$ have mean $0$,

$$\text{cov}(X,Y) = E[XY]$$

Covariance can be normalized to produce the following useful metric.

$$corr(X,Y) = \frac{cov(X,Y)}{\sqrt{V(X)V(Y)}}$$

where 

$$-1 \leq corr(X,Y) \leq 1$$


#### :x Independent properties

If and only if $X$ and $Y$ are independent, the following properties are true:
	
- $E[XY] = E[X]E[Y]$

- $f_{Y|X}(y|x) = f_Y(y) \quad \forall x$

- $V(X + Y + Z + \ldots) = V(X)+V(Y)+V(Z)+\ldots$

### Common distributions

Familiarity with the following distributions is helpful.

<div class="wide">
<div style="min-width:800px">

| | Poisson | Binomial | Normal | Multivariate normal | $\chi_k^2$ |
| --- | --- | --- | --- | --- | --- |
| Parameters | $\lambda$ | $(n,p)$ | $(\mu,\sigma^2)$ | $(\mu,\Sigma)\in(\mathbb{R}^n,\mathbb{R}^{n\times n})$ | $k$ |
| PDF / PMF | $\frac{\lambda^xe^{-\lambda}}{x!}$ | $\frac{n!}{x!(n-x)!}p^x(1-p)^{n-x}$ | $\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\frac{-(x-\mu)^2}{2\sigma^2}}$ | $\frac{\exp(-(x-\mu)^T\Sigma^{-1}(x-\mu)/2)}{(2\pi)^{n/2}\sqrt{\text{det}(\Sigma)}}$ | |
| Mean | $\lambda$ | $np$ | $\mu$ | $\mu$ | $k$ |
| Variance | $\lambda$ | $np(1-p)$ | $\sigma^2$ | $\Sigma$ | $2k$ |
| Minimum sufficient<br/>statistic for n iid | $\Sigma X_i$ | $X$ | $\bar X \to \mu$ | | |
| Notes | | "Bernoulli" when $n=1$. | | | |
</div>
</div>

#### :x Bernoulli distribution

#### :x Poisson distribution

#### :x Uniform distribution

#### :x Normal distribution

#### :x Multi-variate normal distribution
TODO


### Limits, convergence, and asymptotics
We are interested in the behavior of a sequence of random variables $X_1, X_2, ..., X_n$ as $n \to \infty$. Is $X_\infty$ the "same" as some, perhaps easier to compute, $Y$? What approximations can we make for large $n$? 

To answer this question we rely on the concepts of limits and their epsilon-delta formulation. Recall the $\epsilon-\delta$ definition of $\text{lim}_{n\to \infty}x_n=L$ is:
$$\forall\epsilon>0,\exists N>0,s.t.,\forall n\geq N,|x_n-L|<\epsilon$$
For random variables, the concept is similar except the limit is on the *probability* of a deviation. $X_n$ **converges in probability** to $X$ (as $n$ grows large), written $X_n \to_P X$, if:
$$\forall C>0,\forall\epsilon,\exists N>0,s.t.,\forall n\geq N,P\{|X_n-X|>C\}<\epsilon$$
In other words, as $n\to\infty$, the probability distribution of the random variable $X_n - X$ must collapse onto the point $0$. Notice how $X_n$ and $X$ must exist over the same probability space. It would be meaningless to subtract the expected outcome of a die roll with the expected temperature tomorrow and say that they converge.

There is also a weaker concept of convergence. **Convergence in distribution** occurs when the cdf of $X_n$ and $X$ line up as $n\to\infty$. Formally, $X_n \Rightarrow X$ if $\forall x ,\lim_{n \to \infty} F_{X_n}(x) = F_X(x)$. Notice how the definition doesn't depend on random variables, it's just your typical limit. Convergence in probability implies the weaker convergence in distribution, but not vice versa.

The **continuous mapping theorem** (CMT) says that if $X_n$ converges to $X$, then $g(X_n)$ converges to $g(X)$ (applies to both types of convergence).

The **Slutsky theorem** says:
- If, $X_n\to_p X$ and $Y_n \to_p Y$, then $X_n + Y_n\to_p X+Y$ and $X_nY_n \to_p XY$.
- If, $X_n\Rightarrow X$ and $Y_n \to_p c$, then $X_n + Y_n\Rightarrow X+c$ and $X_nY_n \Rightarrow cX$.

**O-notation** is a helpful way of describing asymptotic behavior. $o_p(b_n)$ implies that $b_n$ grows asymptotically *faster than* $X_n$, or formally, $X_n / b_n \to_p 0$. $O_p(b_n)$ means that $b_n$ grows *at least as fast* as $X_n$, or formally:
$$\exists C,\forall\epsilon>0,\exists N>0,s.t.,\forall n\geq N,P\{|X_n/b_n|>C\}<\epsilon$$
Notice how this is only slightly different from the definition of convergence in probability and the stricter $o_p(b_n)$ bound. Some relevant properties:
- Intuitively, $O_p(1)$ means a series is stochastically bounded and $o_p(1)$ means a series approaches $0$ (stochastically).
- $X_n \Rightarrow N(\mu,\sigma^2)$, then $X_n=O_p(1)$.
- $o_p(c \times b_n) = o_p(b_n)$
- $O_p(n^{-\delta}) \to o_p(1) \quad (\delta>0)$
- $o_p(b_n) \to O_p(b_n)$
- $O_p(n^\alpha) O_p(n^\beta) \to O_p(n^{\alpha+\beta})$
- $O_p(n^\alpha) o_p(n^\beta) \to o_p(n^{\alpha+\beta})$
- $O_p(n^\alpha) + O_p(n^\beta) \to O_p(max\{n^\alpha,n^\beta\})$
- $O_p(n^\alpha) + o_p(n^\alpha) \to O_p(n^\alpha)$

Some additional useful rules:
- For *non-negative* random variable $X$, the likelihood of tail events is capped by Markov's theorem, $P\{X\geq t\}\leq E[X]/t$.
- For all random variables, the likelihood of tail events is capped by Chebyskev's inequality, $P\{|X-\mu| \geq t\} \leq Var(X) / t^2$
- Holders inequality todo
- **Law of large numbers** (LLN): For iid random variables $X_1, X_2, \ldots$, with finite variance their average $\bar{X_n} \to_p E[X_i]$.
- **Central limit theorem** (defines *rate* of convergence). That is $\bar{X_n}'\sqrt{n} \Rightarrow N(0,\sigma^2)$. Which implies $\bar{X_n}=O(1/\sqrt{n})$. (The $'$ indicates that we've normalized)

The **delta method** calculates the rate of convergence $g(X)$ given that of $X$. If $\sqrt{n}(X_n-\mu)\Rightarrow N(0,\sigma^2)$ and $g'(\mu) \neq 0$, then $\sqrt{n}(g(X_n) - g(\mu)) \Rightarrow N(0, \sigma^2 g'(\mu)^2)$. It's multivariate extension is, if $\sqrt{n}(\vec{X_n}-\vec{\mu})\Rightarrow N(0,\Sigma)$ and $g: \mathbb{R}^k \to \mathbb{R}$ is twice differentiable, then $\sqrt{n}(g(X_n) - g(\mu)) \Rightarrow N(0, \nabla g (\mu)^T \Sigma \nabla g (\mu))$. 
## Part 2: Statistics
Much like Plato's cave, the philosophy of frequentist statistics is that there exists an ideal world that we cannot observe directly. Our real world is a sample of this ideal world from which me makes inductions. Rather than proving statements deductively (like in most of mathematics), we will use our observations to make inductions about our ideal unobservable world. 

A **statistic** $T(X)$ is any function of our sample $X = (X_1, \ldots, X_n)$. A statistic *is* a random variable because, until we actually take the sample and calculate the statistic, there is a distribution of results we might expect: the **sampling distribution**.

Some statistics are used to estimate parameters of our ideal world. We call these statistics **estimators** and typically denote them with a hat (i.e. estimator $\hat \theta$ for parameter $\theta$).

An estimator is **unbiased** if its expectation equals the parameter (for all $n$). For example, given a sample $X_1, \ldots, X_n$ of i.i.d random variables, the following two statistics are unbiased estimators of parameters $\mu$ and $\sigma^2$.
$$\bar{X_n} = \frac{1}{n} \sum_{i=1}^n X_i$$$$s^2 = \frac{1}{n-1} \sum_{i=1}^n(X_i - \bar{X})^2$$
Note: Why should the unbiased estimator of $\sigma^2$ contain a $n-1$ term? Simplify because this is the estimator that ensures that $E[s^2] =\sigma^2$. If this mathematical result seems surprising, consider that $s^2$ depends on the statistic $\bar{X}$ which will be close, but probably not equal to $\mu$. As such, our sample variance $s^2$ is smaller than what it would be if it were calculated using the real (but unknown) average $\mu$. This means $s^2$ systematically underestimates the real $\sigma^2$ if it weren't for the $n-1$ correction. In other words, the use of $\bar X$ instead of $\mu$ causes $s^2$ to overfit our data and the $n-1$ term corrects for this overfitting.

**Glivenko-Cantelli theorem** says that an empirical cdf $\hat{F}_n(x)$ based on a sample $X_1, \ldots, X_n$ converges on the unobservable distribution $F$. $$\text{sup}_{x\in \mathbb{R}} |\hat F_n(x)-F(x)| \to_p 0$$
There are 4 techniques to find the distribution of a statistic:
1. Analytically - exact but hard for most distributions
2. Monte carlo (by simulation) - only works if you know the unobservable distribution
3. Asymptotic approximation - using the CTL, delta method, slutsky theorem to show that for large $n$ the statistic approaches a given distribution
4. Non-parametric bootstrap - same as monte carlo but use the emprical (sample) distribution instead of the unknown ideal distribution. Works for large $n$ due to glivenko-cantelli theorem.

A plugin estimator is the estimator that uses the same formula as that for the parameter, except using the empirical distribution instead.

If $X_1, \ldots, X_n$ are iid random normal variables then $\bar X_n$ and $s_n^2$ are independent, $\bar{X}_n \sim n(\mu, \sigma^2 / n)$, and $s^2 \sim \sigma^2 \chi^2_{n-1} / (n-1)$.
### Sufficient statistic
Many distributions can be parameterized. For example the standard normal distribution is just one instance of the normal distribution family whose pdf can be written as $f(x|\mu,\sigma^2)$. This is the concept of a **parametric family**. Consider distribution $f(x | \theta)$ with $\theta \in \Theta$, some parametric family.

A statistic $T(X)$ is said **sufficient** (for $\theta$) if our sample $X=(X_1, \ldots, X_n)$ given a specific value of $T(X)$ does not provide any more information on $\theta$ than the value of $T(X)$ alone. More formally, statistic $T(X)$ is sufficient if $X$ conditional on $T(X)$ does not depend on $\theta$. Recall: $$f_{X|T(X)}\left(\ x\ |\ T(X) = T(x)\ \right)= \frac{f_{X}(x)}{f_T(T(x))} $$
For independent variables the joint distribution $f_X(x)= \Pi_{i=1}^n f_{X_i}(x)$

One way to prove that a statistic is sufficient is simply to derive the analytical expression for $f_{X|T(X)}$ and show that it does not contain $\theta$. However, to do this one must already know $T(X)$. Another approach that doesn't require knowing $T(X)$ ahead of time is the factorization theorem.

The **factorization theorem** says that $T(x)$ is sufficient if and only the pdf (or pmf) can be expressed as: $$f(x|\theta)=g(T(x),\theta)\cdot h(x)$$
Notice how the pdf/pmf has been split into two parts: $g$ depends only on the statistic and parameter while $h$ depends only on the sample data. The ability to decompose the pdf/pmf as such indicates that $T$ is a sufficient statistic.

Note that $T$ could actually be several statistics (a vector of statistics if you like) that, combined, capture the information in $\theta$ (or more parameters). There is not always one sufficient statistic per parameter. (For example, estimating $\theta$ in $U[\theta, 1+\theta]$ involves two sufficient statistics $\text{min}_{x_i}$ and $\text{max}_{x_i}$!)

Technically, the entire sample $X_n$ is a sufficient statistic of $X$ (with $n$ terms) but this is not very helpful. A **minimal sufficient statistic** is one that contains all the information on the parameters while being as "small" as possible. That given a sufficient statistic $T^*(X)$ we can say it is minimal if it can be computed from any other sufficient statistic (i.e. $r$ exists and $T^*(X) = r(T(X))$). If this weren't possible, this means that there's another sufficient statistic with less information than $T^*(X)$ so $T^*(X)$ is not minimal. 

The **efficiency** of an estimator is measured using the mean-squared error: $MSE = E[(\hat \theta - \theta)^2]$. A useful property is $MSE(\hat \theta)=Var(\hat \theta) + [\text{Bias}(\hat \theta)]^2$. (proof by expansion and de-meaning)

The **Rao-Blackwell theorem** proves that for any estimator $\hat \theta$ of $\theta$ that depends not only on the sufficient statistic $T(X)$, a just as good estimator $\hat \theta_2 = E(\hat \theta | T(X) )$ exists and depends only on $T(X)$. Here, just as good means that $MSE(\hat \theta_2) \leq MSE(\hat \theta)$ and $\hat \theta_2$ is unbiased if $\hat \theta$ was. The intuition behind this theorem is that an estimator should provide one value per $T(X)$ because any more variation is noise. If it doesn't, the estimator can be improved by averaging across all values with the same $T(X)$ and thus removing this noise.

This theorem allows for efficient Monte-Carlo simulations. Simply draw $B$ samples of size $n$ each. Compute $\hat \theta$ and $T(X)$ for each sample. Now average all the $\hat \theta$ that originated from samples with the same $T(X)$ to obtain an estimate $\hat \theta_2$ of $\theta$ for a given statistic $T(X)$.

### Generating statistics

Sometimes there is no unbiased estimator (see notes for proof). In these cases, the **bootstrap bias corrector** can help produce a less biased corrector: Sample $b \times n$ times from our empirical observations. For every $b$ calculate the "real" parameter value, average across all $b$, then subtract out the estimator to get an estimate of the systemic bias. This estimate can be removed from the estimator to make a less biased one. (This technique removes the O(1/n) bias term, but not smaller biases. It could also be calculated by analytically calculating the bias.)

**Bias-variance tradeoff** is the notion that there's always a more efficient (lower MSE) estimator than the unbiased one (proof in notes).

**Consistency** is the concept that an estimator $\hat \theta_n$ is unbiased as $n \to \infty$. Formally: $\hat\theta_n \to_p \theta$.

An estimator $\hat\theta$ is **asymptotically normal** if $r_n(\hat\theta - a_n) \Rightarrow N(0,\sigma^2)$. Typically, $r_n=\sqrt{n}$ and $a_n = \theta$.

Beyond the plug in estimator (aka. method of analogy) and the **method of moments** (see notes), one can generate the **maximum likelihood estimator**: the estimator that maximizes the likelihood of observing $X$ if $\theta$ were equal to the estimator. This MLE estimator can be found by solving the first order conditions of the likelihood function $L$ (or the log of the likelihood function, aka. log-likelihood $l$). The likelihood function is simply the joint probability distribution.

### Efficient estimators
Note: The following section only applies to parametric families where the support of the distribution (i.e. the "domain", where $f(x) \neq 0$) is independent of $\theta$.

What is the most efficient estimator? The **rao-cramer bound** provides a partial answer as it says that under certain conditions (see below) the variance of $\hat\theta$ has a lower bound: $$Var(\hat\theta)\geq\left(\frac{d}{d\theta}E_\theta[\hat\theta(X)]\right)^2\frac{1}{I(\theta)}$$
where $I(\theta)$ is the fisher information: $$I(\theta)=E_\theta\left[S(\theta\mid X)^2\right]$$
And $S$ is the score $$S(\theta \mid x) = \frac{\partial }{\partial \theta}\log f(x\mid\theta)$$
A special case of the Rao-Cramer bound is for unbiased estimators where $E[\hat\theta]=\theta$ so: $$Var(\hat\theta)\geq\frac{1}{I(\theta)}$$
What does all this mean? Less fisher information means there must be more variance. Our MLE estimator by definition has a score of $0$

For an n-dimensional $\theta$, there are two ways to calculate the $n\times n$ Fisher Information matrix.
1. Calculate the score ($n\times 1$) by taking the derivative of the log likelihood function. Then compute $I=E[S \cdot S^T]$
2. Compute $I$ by taking the second derivatives of the log likelihood and using the **second information equality**: $$I = - E \left[ \frac{\partial ^2 l}{\partial \theta ^2}\right]$$
The MLE estimator is both consistent and asymptotically normal with variance approaching the Rao-Cramer bound.

## Part 3: Statistical tests