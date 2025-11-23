---
title: "Class notes from MIT's 14.380 \"Statistics for Economics\" course"
description: Post carefully explaining statistical concepts from class. 
date: 2025-11-22
tags: communication
draft: false
---

_These are part of my class notes for the MIT course 14.380 Statistics for Economics that I took in Fall 2025. I wrote up these notes to experiment with Nutshells, expandable boxes [:like this one](#x-nutshells) that should make my notes more accessible to different audiences and easier to navigate. My blog post [here](TODO) provides more context. Parts of my notes may be inaccurate or poorly explained, especially the details within Nutshells that I've paid less attention to._

#### :x Nutshells

Clicking on an expandable box like you just did will reveal additional details that I didn't want to include in the main flow of the text. I'm hoping you like this feature because I spent lots of time getting it to work! Thanks to the fantastic creator Nicky Case for inventing these (see the [nutshell project](https://ncase.me/nutshell/))!

## Course overview
This course covers the mathematical foundations in statistics needed for doing PhD-level economics and econometrics. It is organized as follows:
1. [Part 1](#part-1-probability) covers the foundations of statistics, namely random variables and tools to analyze their behaviors as $n \to \infty$, including the delta method, the central limit theorem and O-notation. These tools will be useful to estimate our statistics for large sample sizes.
2. [Part 2](#part-2-statistics-and-estimators) covers statistics and estimators, particularly what makes a "good" estimator and how to estimate "goodness" through tools like the Rao-Cramer bound and Fisher information.
3. Part 3 covers statistical tests. I have not written up my notes for this part.

Familiarity with calculus, matrix algebra and other math concepts like the [:taylor series expansion](#x-taylor-series) is important.

#### :x Taylor Series

Proofs often rely on approximating $f(x) - f(a)$ using the [Taylor Series](https://en.wikipedia.org/wiki/Taylor_series) expansion:

$$f(x)-f(a)=\frac{f'(a)}{1!}(x-a)+\frac{f''(a)}{2!}(x-a)^2+\ldots$$

When a finite number of terms are used, the approximation is most accurate near $a$.

## Part 1: Probability

It is often said that probability is the inverse of statistics[^1]. While statistics is about making inferences about the world given observations, probability is about making claims as to what me will observe given a predefined world. Statistics is inductive. Probability is deductive. Statistics is backward looking. Probability is forward looking.

[^1]: See for example this [StackExchange answer](https://stats.stackexchange.com/a/675).

Although technically correct, I don't find this definition particularly helpful. Rather, I like to think of probability as the field of pure mathematics that serves as the foundation of statistics, a field of applied mathematics. Probability tells you what to expect given a fully-defined mathematical problem. Statistics suggests how one can convert the real world into such a problem and meaningfully interpret the problem's results. Probability is entirely built upon a [:set of mathematical axioms](#x-probability-spaces). Statistics is built on probability with an added sprinkle of _philosophy_.

For example, frequentist statistics (the focus of this course) adopts of philosophical framework reminiscent of Plato's cave. It assumes that there exists a real world containing the truth we wish to find, but we cannot directly observe this real world. Rather, we only get to observe samples drawn randomly from this real world from which we are to make inductions about the real world (aided by tools from probability). Bayesian statistics is another approach to statistics grounded in a different philosophy. In general, the application of probability to the real world is a contested and fascinating area of philosophy[^2].

[^2]: For example, see this [cheeky paper](https://statistics.berkeley.edu/sites/default/files/tech-reports/611.pdf) or, for those with access, read: Freedman, D. Some issues in the foundation of statistics. Found Sci 1, 19–39 (1995). https://doi.org/10.1007/BF00208723

We're now ready to dive into probability, the mathematical framework that will underpin the statistics discussed in Parts 2 and 3.

#### :x Probability spaces

Formally, probability is a subset of measure theory, a branch of mathematics that helps deal with measurable sets such as magnitudes, masses, geometrical measures like length, area, volume, and, of course, probability.

From a measure theory perspective, all probabilities originate from a **probability space**, a triplet of three mathematical objects: $(\Omega, \mathcal{F}, P)$. $\Omega$ is the **sample space** which represents all potential outcomes (realizations) of an experiment (i.e. a single draw). Note that these realizations $\omega$ need not be a number; $\omega$ could be the description "the first die rolled a 5 and the second rolled a 4." Random variables are functions that map these realizations $\omega$ to the space of real numbers $\mathbb{R}$. In order for random variables to have a cumulative probability distribution $F_X(t): \mathbb{R} \to [0,1]$, measure theory introduces the concept of a **probability measure** $P$, a function that maps _measurable sets of outcomes_ $\mathcal{F}$ to the probability of observing an outcome in that set. Note the language: although conceptually you can think of a probability measure $P$ as simply mapping outcomes $\omega \in \Omega$ to the probability $[0,1]$ of that outcome, it is actually mapping all _measurable_ sets of outcomes $\mathcal{F}$ to their probabilities. While beyond the scope of this course, $\mathcal{F}$ is necessary because not all sets of $\Omega$ are measurable (see this [video on Vitali sets](https://www.youtube.com/watch?v=hs3eDa3_DzU)).

To summarize, measure theory formalizes probability as follows:
- Probabilities originate from a probability space $(\Omega, \mathcal{F}, P)$. 
- $\Omega$ is the sample space, the space of all potential experimental outcomes.
- $\omega \in \Omega$ is any one sample or outcome.
- Random variables are functions that map $\Omega \to \mathbb{R}$.
- $\mathcal{F}$ is the space of all measurable subsets of $\Omega$ (i.e. events).
- The probability measure $P$ maps $\mathcal{F} \to [0,1]$.
- The cumulative probability distribution $F_X$ stems from combining the probability measure $P$ and the random variable $X$.

To learn more, read about [sigma algebras](https://en.wikipedia.org/wiki/%CE%A3-algebra) and [measure theory](https://en.wikipedia.org/wiki/Measure_(mathematics)).

### Random variables

A **random variable** (r.v.) is a mathematical object *dependent on random events*. Examples include:
- A random variable representing the result of a coin toss ($1$ for heads, $0$ for tails).
- A random variable representing the average height of the next 10 people you see.
- A random variable representing the temperature tomorrow.

Importantly, random variables have a probability distribution. $X \sim F_x$ indicates that $X$ is a random variable with cumulative probability distribution (cdf) $F_x(t) = P \{X \leq t\}$. Key tools to work with random variables include the [:probability density function](#x-def-pdf) (pdf) $f_X(t)$, the [:expected value function](#x-def-expected-value) $E[X]$, [:variance](#x-variance) $V(X)$, [:moments](#x-moments), and [:important properties](#x-properties-of-random-variables) of these abstractions.

Common probability distributions include the [:binomial distribution](#x-binomial-distribution), [:poisson and gamma distribution](#x-poisson-distribution), [:uniform distribution](#x-uniform-distribution), and [:normal distribution](#x-normal-distribution).

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

The expected value function is linear,
$$E[X+Y] = E[X] + E[Y]$$
$$E[aX] = aE[X]$$

But variance is not: 

$$Var(X+Y) \neq Var(X) + Var(Y)$$

Rather,
	- $Var(X+a)=Var(X)$ (shift agnostic)
	- $Var(aX)=a^2Var(X)$

An often used property is:
$$Var(X) =E[X^2] - E[X]^2$$

Note that when $X$ has mean 0, $Var(X) = E[X^2]$.

#### :x Binomial distribution

A binomial distribution models the number of successes $X$ in $n$ independent "coin tosses" with success probability $p$.

$$P\{X=x\} = \frac{n!}{x!(n-x)!}p^x(1-p)^{n-x}$$

It has mean $\mu = np$, variance $\sigma^2 = np(1-p)$.

When $n=1$, the binomial distribution is called a Bernoulli distribution. This is simply a coin toss:

$$P\{X=1\} = p, \quad P\{X=0\} = 1-p$$

#### :x Poisson distribution

If the average rate of events is $\lambda$ per interval (e.g. 1000 customers per year) and events occur independently, then the probability of observing $x$ events in one interval is given by the Poisson distribution:

$$P\{X=x\} = \frac{\lambda^x e^{-\lambda}}{x!}$$

Its mean and variance equal $\lambda$ and the minimum sufficient statistic for $\lambda$ is $\sum_{i=1}^n X_i$.

Related to the Poisson distribution is the Gamma distribution which models the waiting time for the $k$-th event to occur:

$$f_X(x) = \frac{\lambda^k x^{k-1} e^{-\lambda x}}{(k-1)!}$$

When we're only interested in the waiting time for the first event ($k=1$), the Gamma distribution simplifies to the exponential distribution:

$$f_X(x) = \lambda e^{-\lambda x}$$

#### :x Uniform distribution

A uniform distribution on the interval $[a,b]$ is one where all values between $a$ and $b$ are equally likely,

$$f_X(x) = \frac{1}{b-a} \quad \text{for } a \leq x \leq b$$

It has mean $\mu = \frac{a+b}{2}$ and variance $\sigma^2 = \frac{(b-a)^2}{12}$.

#### :x Normal distribution

Normal distributions arise from the central limit theorem and are thus ubiquitous in statistics. A normal distribution with mean $\mu$ and variance $\sigma^2$ has pdf:

$$f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp{\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}$$

If $X$ is a vector of $n$ normal random variables with mean vector $\mu$ and covariance matrix $\Sigma$, then $X$ has a multivariate normal distribution with pdf:

$$f_X(x) = \frac{\exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)}{(2\pi)^{n/2}\sqrt{\text{det}(\Sigma)}}$$

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

1. $Var(X+Y)=Var(X)+Var(Y)+2\text{cov}(X,Y)$
2. $\text{cov}(X,Y) = E[XY] - E[X]E[Y]$

Notice that if $X$ or $Y$ have mean $0$,

$$\text{cov}(X,Y) = E[XY]$$

Covariance can be normalized to produce the following useful metric.

$$corr(X,Y) = \frac{cov(X,Y)}{\sqrt{V(X)V(Y)}}$$

where by definition 

$$-1 \leq corr(X,Y) \leq 1$$

[:Proof](#x-corr-proof)

##### :x Corr Proof

Let $A = X - E[X]$ and $B = Y - E[Y]$. From [:Holder's inequality](#x-holders-inequality) we know:

$$E[AB]^2 \leq E[A^2]E[B^2]$$

Substituting in our definitions for $A$ and $B$ we find:

$$cov(X,Y)^2 \leq V(X) V(Y)$$

Therefore, by rearranging,

$$\frac{|cov(X,Y)|}{\sqrt{V(X) V(Y)}} \leq 1$$

#### :x Independent properties

If and only if $X$ and $Y$ are independent, the following properties are true:
	
- $E[XY] = E[X]E[Y]$

- $f_{Y|X}(y|x) = f_Y(y) \quad \forall x$

- $V(X + Y + Z + \ldots) = V(X)+V(Y)+V(Z)+\ldots$


### Limits, convergence, and asymptotics

Later in the course, we will want to make approximations for large sample sizes. For example, we might want to argue that some random variable of our current sample $X_n$ is actually well approximated by the random variable $X$ when $n$ is large. Such arguments can simplify our statistics since often $X$ is a simpler expression to work with than $X_n$.

To make such arguments, we must introduce mathematical tools to deal with _asymptotics_—the behavior of random variables as $n$ approaches infinity (denoted $n \to \infty$).

You may be familiar with limits and their [:delta-epsilon formalization](#x-delta-epsilon-formalization) to describe the behavior of (non-random) numbers that approach infinity. An extension of this same concept exists for random variables.

Specifically, there are two notable ways in which a random variable $X_n$ can be said to converge onto a different random variable $Y$. 

1. **Convergence in distribution** (denoted $X_n \stackrel{d}{\to}Y$) occurs when the probability distribution of $X_n$ approaches that of $Y$ (the CDFs "line up"). Distribution-wise $X_n$ and $Y$ are identical in the limit. However, the _outcomes_ of a draw of $X_n$ and $Y$ need not be identical. For example, say you have two magic coins that always land on opposite faces. (When you toss them simultaneously one always lands head and the other tail.) The distribution of both coins are equal (50% heads, 50% tails) but their realizations are not.

2. **Convergence in probability** (denoted $X_n \stackrel{p}{\to}Y$) is the stronger idea that not only do the distributions of $X_n$ and $Y$ match in the limit, but also the _realizations_ of any individual draws match (in the limit). Of course, for this concept to make any sense, $X_n$ and $Y$ must come from the same draw (i.e. the same [:probability space](#x-probability-spaces)). (If the magic coins can be tossed separately you cannot talk about comparing their outcomes, only their distributions.)

Both types of convergence have [:formal definitions and properties](#x-definitions-of-convergence) that are useful to know (e.g. convergence in probability implies convergence in distribution).

Now that we are equipped to discuss asymptotic behaviors we can introduce several useful tools:

- The **law of large numbers** (LLN) states that the average of $n$ independent and identically distributed (iid) random variables converges in probability to their expected value: $\bar{X_n} \stackrel{p}{\to} E[X_i]$.

- But how fast does an average converge? The **central limit theorem** says that the variance of an average shrinks at rate $\frac{1}{\sqrt{n}}$. You can think of this as saying $\bar{X_n} \stackrel{d}{\to}N(E[X_i], \frac{\sigma^2}{n})$ although technically we cannot converge onto a distribution that itself is changing with $n$ and must instead write the CTL as:

$$\sqrt{n}(\bar{X_n}-\mu_X) \stackrel{d}{\to} N(0,\sigma^2)$$

- [:**O-notation**](#x-o-notation) helps us make arguments about _rates_ of convergence.

- The [:**Continuous Mapping Theorem**](#x-cmt) (CMT), [:**Delta Method**](#x-delta-method), and [:**Slutsky Theorem**](#x-slutsky-theorem) help derive the asymptotic behaviors of functions of random variables (e.g. $g(X)\to ?$ or $X+Y \to?$)

- Finally the following inequalities are useful in setting bounds on tail events: [:Chebyskev's inequality](#x-chebyskevs-inequality), [:Markov's theorem](#x-markovs-theorem),and [:Hölder's inequality](#x-holders-inequality).

#### :x CMT

What are the properties of _functions of random variables_? The continuous mapping theorem (CMT) says that if $X_n$ converges to $X$ in probability/distribution, then $g(X_n)$ converges to $g(X)$ in probability/distribution.

#### :x Delta method

The delta method calculates the rate of convergence $g(X)$ given that of $X$. If $\sqrt{n}(X_n-\mu)\stackrel{d}{\to} N(0,\sigma^2)$ and $g'(\mu) \neq 0$, then $\sqrt{n}(g(X_n) - g(\mu)) \stackrel{d}{\to} N(0, \sigma^2 g'(\mu)^2)$. It's multivariate extension is, if $\sqrt{n}(\vec{X_n}-\vec{\mu})\stackrel{d}{\to} N(0,\Sigma)$ and $g: \mathbb{R}^k \to \mathbb{R}$ is twice differentiable, then $\sqrt{n}(g(X_n) - g(\mu)) \stackrel{d}{\to} N(0, \nabla g (\mu)^T \Sigma \nabla g (\mu))$.

#### :x Slutsky theorem

The Slutsky theorem says that sum or product of two variables that converge in probability also converges in probability (if, $X_n\stackrel{p}{\to} X$ and $Y_n \stackrel{p}{\to} Y$, then $X_n + Y_n\stackrel{p}{\to} X+Y$ and $X_nY_n \stackrel{p}{\to} XY$). While this is not true for two variables that converge in distribution, we can say that if $X_n\stackrel{d}{\to} X$ and $Y_n \stackrel{p}{\to} c$, then $X_n + Y_n\stackrel{d}{\to} X+c$ and $X_nY_n \stackrel{d}{\to} cX$.


#### :x Chebyskev's inequality

Chebyskev's inequality caps the probability of tail events. For any random variable $X$,

$$P\{|X-\mu| \geq t\} \leq \frac{Var(X)}{t^2}$$

#### :x Markov's theorem

Like [:Chebyskev's inequality](#x-chebyskevs-inequality), Markov's theorem sets an upper bound on the probability of tail events, but the theorem is specific to *non-negative* random variables only. For non-negative random variable $X$,

$$P\{X\geq t\}\leq \frac{E[X]}{t}$$

#### :x Holder's inequality

For all $p,q>1$ such that $\frac{1}{p}+\frac{1}{q}=1$,

$$E[|XY|] \leq E[|X|^p]^{\frac{1}{p}}E[|Y|^q]^{\frac{1}{q}}$$

Note that the Cauchy-Schwarz inequality is the special case where $p=q=2$,

$$E[XY]^2 \leq E[X^2]E[Y^2]$$

#### :x Delta-epsilon formalization

Formally, function $f(n)$ is said to approach some number $L$ in the limit $n \to \infty$, if, given any distance $\epsilon$, we can always find a threshold $N$ where for all $n > N$ such that the result $f(n)$ is within distance $\epsilon$ from $L$. Mathematically, $\lim_{n\to\infty} f(n) = L$ means that

$$\forall \epsilon > 0, \exists N > 0 \text{ such that } \forall n \geq N, |f(n) - L| < \epsilon$$

#### :x Definitions of convergence

$X_n$ is said to converge _in distribution_ onto $Y$ (denoted $X_n \stackrel{d}{\to}Y$) if:

$$\forall x, \lim_{n\to\infty} F_{X_n}(x) = F_Y(x)$$

Or, using delta-epsilon notation:

$$\forall x,\forall \epsilon>0, \exists N, \text{ such that }\forall n>N, |F_{X_n}(x)-F_Y(x)|<\epsilon$$

$X_n$ is said to converge _in probability_ onto $Y$ (denoted $X_n \stackrel{p}{\to}Y$) if the likelihood of a significant deviation in any one draw ($P\{|X_n - Y|>C\}$) approaches zero. Using delta-epsilon notation:

$$\forall C>0,\forall\epsilon>0,\exists N,s.t.,\forall n\geq N,P\{|X_n-Y|>C\}<\epsilon$$

In other words, as $n\to\infty$, the probability distribution of the random variable $X_n - Y$ must collapse onto the point $0$. Again, notice how $X_n$ and $Y$ must exist over the same probability space. It would be meaningless to subtract the expected outcome of a die roll with that of a different die roll unless the dies were always rolled simultaneously.

#### :x O-notation

A convenient way to deal with _rates_ of convergence is using **O-notation**. We write $X_n \in o(b_n)$ if the spread of $X_n$ shrinks faster than $b_n$ and $X_n \in O(b_n)$ if the spread of $X_n$ shrinks at the same speed (or faster) than $b_n$. For example, the CLT implies than the variance of an average shrinks at speed $1/\sqrt{n}$, so $\bar X_n \in O(n^{-1/2})$. We could also write $o_p(1)$ to simply denote that the variance of an average approaches zero. O-notation allows us to more easily make arguments about rates of convergence.

Familiarity with the formal definitions and properties of O-notation (below) is helpful.

Formally, we say $X_n \in o_p(b_n)$ if $X_n / b_n \stackrel{p}{\to} 0$ which in delta-epsilon notation is:

$$\forall C>0,\forall\epsilon>0,\exists N,s.t.,\forall n\geq N,P\{|X_n/b_n|>C\}<\epsilon$$


We say, $X_n\in O_p(b_n)$ if 

$$\exists C,\forall\epsilon>0,\exists N,s.t.,\forall n\geq N,P\{|X_n/b_n|>C\}<\epsilon$$

Notice the only slight difference between both definitions and how $o_p(b_n)$ is stricter than $O_p(b_n)$. 


Some helpful properties of o-notation:
- Intuitively, $O_p(1)$ means a series is stochastically bounded (doesn't grow to infinity) and $o_p(1)$ means a series approaches $0$ (stochastically).
- $X_n \stackrel{d}{\to} N(\mu,\sigma^2)$, then $X_n=O_p(1)$.
- $o_p(c \times b_n) = o_p(b_n)$
- $O_p(n^{-\delta}) \to o_p(1) \quad (\delta>0)$
- $o_p(b_n) \to O_p(b_n)$
- $O_p(n^\alpha) O_p(n^\beta) \to O_p(n^{\alpha+\beta})$
- $O_p(n^\alpha) o_p(n^\beta) \to o_p(n^{\alpha+\beta})$
- $O_p(n^\alpha) + O_p(n^\beta) \to O_p(max\{n^\alpha,n^\beta\})$
- $O_p(n^\alpha) + o_p(n^\alpha) \to O_p(n^\alpha)$

## Part 2: Statistics and estimators

Now that we've covered the mathematical tools of probability we are ready to discuss statistics. Recall that, much like [Plato's cave](https://en.wikipedia.org/wiki/Allegory_of_the_cave), the philosophy of frequentist statistics is that there exists a real world that we cannot observe directly. Rather, we only have a sample of that real world from which me makes inductions. We shall denote this sample as $x = (x_1, \ldots, x_n)$. Before actually observing the sample, $x$ is a random variable since the value of any draw $x_i$ is still uncertain. As such, we will denote the before-observation sample as $X = (X_1, \ldots, X_n)$.

A **statistic** $T(X)$ is any function of our sample $X = (X_1, \ldots, X_n)$. For example, the sample mean $\bar X$ is a statistic. Again, note that a statistic *is* a random variable because, until we actually take the sample and calculate the statistic, there is a distribution of results we might expect: the **sampling distribution**. Since statistics are random variables, we can discuss their properties like $E[\bar X]$ and $Var(\bar X)$.

**Parameters** are properties of the real (unobservable) world. Some statistics are particularly useful to estimate parameters. We call these statistics **estimators** and typically denote them with a hat (i.e. estimator $\hat \theta$ for parameter $\theta$). For example, sample mean $\bar X$ is an estimator of the real average $\mu$ (and thus, $X$ and $\hat\mu$ are often used interchangeably).

What makes a "good" estimator? There are several properties to evaluate the goodness of estimators:

- [:Bias](#x-bias), whether over multiple experiments the average of the estimators converges onto the parameter.
- [:Efficiency](#x-efficiency), a proxy for the spread (variance) of the sampling distribution of the estimator.
- [:Consistency](#x-consistency), whether for large sample sizes, the bias converges to $0$. 
- [:Asymptotic normality](#x-asymptotic-normality), whether the sampling distribution of an estimator converges onto a normal.

Note that there is a **bias-efficiency tradeoff** in that there is always a more efficient estimator than the unbiased one. Moreover, the [:**Rao-Cramer bound**](#rao-cramer-bound) sets a lower bound on the variance of an estimator (this bound is based in the notion of Fisher information, a proxy for how much information the sample can convey about the distribution).

#### :x Bias

An estimator $\hat \theta$ is **unbiased** if $E[\hat \theta] = \theta$. (Recall, $\hat \theta$ is a random variable while $\theta$ isn't.) For example, given a sample $X_1, \ldots, X_n$ of i.i.d random variables, the following two statistics are unbiased estimators of parameters $\mu$ and $\sigma^2$, respectively.

$$\bar{X_n} = \frac{1}{n} \sum_{i=1}^n X_i$$

$$s^2 = \frac{1}{n-1} \sum_{i=1}^n(X_i - \bar{X})^2$$

([:Curious about why $n-1$?](#x-why-n-1))

#### :x Why n-1?
Why should the unbiased estimator of $\sigma^2$ contain a $n-1$ term? Simplify because this is the estimator that ensures that $E[s^2] =\sigma^2$. If this mathematical result seems surprising, consider that $s^2$ depends on the statistic $\bar{X}$ which will be close, but probably not equal to $\mu$. As such, our sample variance $s^2$ is smaller than what it would be if it were calculated using the real (but unknown) average $\mu$. This means $s^2$ systematically underestimates the real $\sigma^2$ if it weren't for the $n-1$ correction. In other words, the use of $\bar X$ instead of $\mu$ causes $s^2$ to overfit our data and the $n-1$ term corrects for this overfitting.

#### :x Sufficient statistic
Many distributions can be parameterized. For example the standard normal distribution is just one instance of the normal distribution family whose pdf can be written as $f(x|\mu,\sigma^2)$. This is the concept of a **parametric family**. Consider distribution $f(x | \theta)$ with $\theta \in \Theta$, some parametric family.

A statistic $T(X)$ is said **sufficient** (for $\theta$) if our sample $X=(X_1, \ldots, X_n)$ given a specific value of $T(X)$ does not provide any more information on $\theta$ than the value of $T(X)$ alone. More formally, statistic $T(X)$ is sufficient if $X$ conditional on $T(X)$ does not depend on $\theta$. Recall: $$f_{X|T(X)}\left(\ x\ |\ T(X) = T(x)\ \right)= \frac{f_{X}(x)}{f_T(T(x))} $$
For independent variables the joint distribution $f_X(x)= \Pi_{i=1}^n f_{X_i}(x)$

One way to prove that a statistic is sufficient is simply to derive the analytical expression for $f_{X|T(X)}$ and show that it does not contain $\theta$. However, to do this one must already know $T(X)$. Another approach that doesn't require knowing $T(X)$ ahead of time is the [:factorization theorem](#x-factorization-theorem).

Note that $T$ could actually be several statistics (a vector of statistics if you like) that, combined, capture the information in $\theta$ (or more parameters). There is not always one sufficient statistic per parameter. (For example, estimating $\theta$ in $U[\theta, 1+\theta]$ involves two sufficient statistics $\text{min}_{x_i}$ and $\text{max}_{x_i}$!)

Technically, the entire sample $X_n$ is a sufficient statistic of $X$ (with $n$ terms) but this is not very helpful. A **minimal sufficient statistic** is one that contains all the information on the parameters while being as "small" as possible. Given a sufficient statistic $T^*(X)$ we can say it is minimal if it can be computed from any other sufficient statistic (i.e. $r$ exists and $T^*(X) = r(T(X))$). If this weren't possible, this means that there's another sufficient statistic with less information than $T^*(X)$ so $T^*(X)$ is not minimal. 

#### :x Rao-Blackwell

The **Rao-Blackwell theorem** proves that for any estimator $\hat \theta$ of $\theta$ that depends not only on the [:sufficient statistic](#x-sufficient-statistic) $T(X)$, a just as good estimator $\hat \theta_2 = E(\hat \theta | T(X) )$ exists and depends only on $T(X)$. Here, just as good means that $MSE(\hat \theta_2) \leq MSE(\hat \theta)$ and $\hat \theta_2$ is unbiased if $\hat \theta$ was. 

The intuition behind this theorem is that an estimator should provide one value per $T(X)$ because any more variation is noise. If it doesn't, the estimator can be improved by averaging across all values with the same $T(X)$ and thus removing this noise.

This theorem allows us to derive an improved estimator $\hat\theta_2$. Although this can sometimes be tricky to do analytically, it is easy to do with a [:Monte-Carlo simulation](#x-monte-carlo). Simply draw $B$ samples of size $n$ each. Compute $\hat \theta$ and $T(X)$ for each sample. Now average all the $\hat \theta$ that originated from samples with the same $T(X)$ to obtain an estimate $\hat \theta_2$ of $\theta$ for a given statistic $T(X)$.

#### :x Efficiency

The **efficiency** of an estimator is measured using the mean-squared error: $MSE = E[(\hat \theta - \theta)^2]$. A useful property is $MSE(\hat \theta)=Var(\hat \theta) + [\text{Bias}(\hat \theta)]^2$. (proof by expansion and de-meaning)

#### :x Consistency

**Consistency** is the concept that an estimator $\hat \theta_n$ is unbiased as $n \to \infty$. Formally: $\hat\theta_n \stackrel{p}{\to} \theta$.

#### :x Asymptotic normality

An estimator $\hat\theta$ is **asymptotically normal** if $r_n(\hat\theta - a_n) \stackrel{d}{\to} N(0,\sigma^2)$. Typically, $r_n=\sqrt{n}$ and $a_n = \theta$.

#### :x Factorization theorem

The **factorization theorem** says that $T(x)$ is a sufficient statistic if and only the pdf (or pmf) can be expressed as: $$f(x|\theta)=g(T(x),\theta)\cdot h(x)$$
Notice how the pdf/pmf has been split into two parts: $g$ depends only on the statistic and parameter while $h$ depends only on the sample data. The ability to decompose the pdf/pmf as such indicates that $T$ is a sufficient statistic.

### Generating estimators and their sampling distributions

How does one _find_ a good estimator?

- Use the plugin estimator: the estimator that uses the same formula as that for the parameter, except using the empirical distribution instead. (e.g. sample mean $\hat\mu=\frac{1}{n}\sum_{i=1}^n X_i$ to approximate $\mu=\int_{-\infty}^\infty xf(x)dx$)

- Use the notion of [:sufficient statistics](#x-sufficient-statistic), a set of tools based on the fact that certain parameters can be entirely "captured" by a set of statistics ("sufficient statistics"). Estimators that only depend on those sufficient statistics are more efficient and less biased (or at least no worse, see [:Rao-Blackwell theorem](#x-rao-blackwell)).

- Derive the [:maximum likelihood estimator](#x-mle) (MLE). MLE estimators are consistent and asymptotically normal with variance approaching the [:Rao-Cramer bound](#rao-cramer-bound).

- Use a [:bootstrap-bias correction](#x-bootstrap-bias-correction), especially useful for the cases where there is no unbiased estimator.

- Use the [method of moments](https://en.wikipedia.org/wiki/Method_of_moments_(statistics)) (not covered in depth in the course).

Once one obtains the formula for an estimator (or statistic), how can one calculate its sampling distribution? There are generally four ways to do so:

1. The distribution of simple estimators can be derived analytically. For example, it can be shown that if $X_1, \ldots, X_n$ are iid random normal variables, the distribution of the sample mean $\bar X$ is the normal $N(\mu, \sigma^2 / n)$ and the distribution of the sample variance $s^2$ is $\sigma^2 \chi^2_{n-1} / (n-1)$.

2. For some estimators, the distribution can be approximated for large $n$ by using asymptotic arguments (e.g. using the CLT, delta method, Slutsky theorem). For example, the central limit theorem says that for large enough $n$, $\bar X \sim N(\mu, \sigma^2/n)$.

3. Via a computer simulation with the [:Monte Carlo method](#x-monte-carlo) (requires knowing or assuming the true distribution).

4. Via a computer simulation with the [:non-parametric bootstrap method](#n-p-bootstrap) (doesn't require knowing the true distribution but requires large enough $n$).

Note that many of these methods require assuming a parametric family for the true distribution (e.g. assuming we're sampling from a Poisson distribution). As such, we might have a good estimator, but if our assumption is poor (aka. model mis-specification), we will be estimating the wrong thing. Specifically, we will be estimating a parameter $\theta_0$ which is "pseudo-true" in the sense that it is a mix of the real and assumed distribution. Whether $\theta_0$ is meaningful depends on the context (e.g. how far off are we), however, the variance of the estimator of $\theta_0$ can be estimated using [:White's standard errors](#x-whites-standard-errors) and this formula for variance is often preferred since it is thought to be more "robust" to model mis-specification.

#### :x White's standard errors

In a famous 1980 paper, White showed that if the assumed distribution is incorrect, the maximum likelihood estimator $\hat \theta_{ML}$ estimates a "pseudo-true" parameter $\theta_0$ in such a way that the MLE is still asymptotically normal except at rate of $\Sigma_2^{-1}\Sigma_1\Sigma_2^{-1}$. Specifically:

$$\sqrt{n}(\hat \theta_{ML} - \theta_0) \stackrel{d}{\to} N(0,\Sigma_2^{-1}\Sigma_1\Sigma_2^{-1})$$

where $\Sigma_1$ and $\Sigma_2$ are both equal to the Fisher information $I(\theta_0)$ except calculated in different ways:

$$\Sigma_1 = E\left[S(X \mid \theta_0)^2\right] \qquad \Sigma_2 -E\left[\frac{\partial^2 l}{\partial\theta_0^2}\right]$$

This formula for estimating the variance is thus preferred since it is considered robust to model mis-specification (only to the extent that the pseudo-true parameter $\theta_0$ remains meaningful).

#### :x Bootstrap bias correction

Sometimes there is no unbiased estimator. In these cases, the bootstrap bias corrector can help produce a less biased estimator: Sample $b \times n$ times from our empirical observations. For every $b$ calculate the "real" parameter value, average across all $b$, then subtract out the estimator to get an estimate of the systemic bias. This estimate can be removed from the estimator to make a less biased one. (This technique removes the O(1/n) bias term, but not smaller biases. It could also be removed by analytically calculating the bias.)

#### :x Monte Carlo

If one knows (or assumes) the unobservable distribution, one can simply simulate drawing a sample of size $n$, calculating the statistic and then repeating $b$ times to get an empirical distribution for the statistic. As $b \to \infty$, the empirical distribution converges on the statistic's sampling distribution. This technique is called a **Monte Carlo technique**.

#### :x N. P. Bootstrap

This technique is the same as the [:monte carlo method](#x-monte-carlo) except instead of sampling from the true distribution, sample repeatedly from the empirical distribution (i.e your observed sample $x$) which according to the [:Gilvenko-Cantelli theorem](#x-gilvenko-cantelli) should yield the same result for large enough $n$.

#### :x Gilvenko-Cantelli

The **Glivenko-Cantelli theorem** says that an empirical cdf $\hat{F}_n(x)$ based on a sample $X_1, \ldots, X_n$ converges on the unobservable distribution $F$. 

$$\text{sup}_{x\in \mathbb{R}} |\hat F_n(x)-F(x)| \stackrel{p}{\to} 0$$

#### :x MLE

The maximum likelihood estimator $\hat \theta_{MLE}$ is the estimator that maximizes the likelihood of observing our sample $X$ if $\theta$ were equal to $\hat \theta_{MLE}$. In other words, it yields the $\theta$ that is most likely to have yielded our sample.

The MLE estimator can be found by solving the first order conditions of the likelihood function $L$ (or the log of the likelihood function, aka. log-likelihood $l$). The likelihood function is simply the joint probability distribution.

#### :x Rao-Cramer bound

Note: This section only applies to parametric families where the support of the distribution (i.e. the "domain", where $f(x) \neq 0$) is independent of $\theta$.

What is the most efficient estimator? The **rao-cramer bound** provides a partial answer as it says that under certain conditions (see below) the variance of $\hat\theta$ has a lower bound: 

$$Var(\hat\theta)\geq\left(\frac{d}{d\theta}E_\theta[\hat\theta(X)]\right)^2\frac{1}{I(\theta)}$$

where $I(\theta)$ is the Fisher information: 

$$I(\theta)=E_\theta\left[S(\theta\mid X)^2\right]$$

And $S$ is the score 

$$S(\theta \mid x) = \frac{\partial }{\partial \theta}\log f(x\mid\theta)$$

A special case of the Rao-Cramer bound is for unbiased estimators where $E[\hat\theta]=\theta$ so: $$Var(\hat\theta)\geq\frac{1}{I(\theta)}$$
What does all this mean? Less Fisher information means there must be more variance.

MLE estimators by definition have an expected score of $0$.

For an n-dimensional $\theta$, there are two ways to calculate the $n\times n$ Fisher Information matrix.
1. Calculate the score ($n\times 1$) by taking the derivative of the log likelihood function. Then compute $I=E[S \cdot S^T]$
2. Compute $I$ by taking the second derivatives of the log likelihood and using the **second information equality**: $$I = - E \left[ \frac{\partial ^2 l}{\partial \theta ^2}\right]$$