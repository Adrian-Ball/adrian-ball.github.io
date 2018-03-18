---
layout: article
title: "Problem 2"
date: 2018-03-18
categories: project-euler
author: adrian-ball
image:
  teaser: project-euler/problem-2/fibonacci-teaser.jpg
  feature: project-euler/problem-2/fibonacci-banner.jpg
---

In [this](https://projecteuler.net/problem=2) problem, the task is to sum all even Fibonacci numbers whose values do not exceed four million. The Fibonacci sequence is a series of numbers where each number in the sequence is the sum of the two previous numbers. Formally we denote this as $$F_n = F_{n-1} + F_{n-2} $$. The start of the sequence is: 

$$ 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, \dots$$

Note that the series is often presented as starting with $$1, 1$$.

The straight-forward, brute-force solution is to check every number in the sequence up to the limit,  and add it to the answer if it meets the requirement of being an even number. This is presented in the code segment below, where we check to see if the number is even, and add it to the answer if so. The rest of the code in the loop updates the three most recent terms, $$F_{n-2}, F_{n-1}, F_{n}$$.

{% highlight python %}

#Initialise these values as the first and second even fibonacci numbers
older_fib_term = 1
recent_fib_term = 2

limit = 4000000 
answer = 2 #First even fibonacci
next_fib_term = older_fib_term + recent_fib_term

while next_fib_term <= limit:
    if next_fib_term % 2 == 0:
        answer += next_fib_term    
    
    older_fib_term = recent_fib_term
    recent_fib_term = next_fib_term
    next_fib_term = older_fib_term + recent_fib_term
    
print(answer)

{% endhighlight %}

While this definitely works and is relatively straightforward, there is some room for improvement. Looking at the initial values of the Fibonacci sequence, it looks like the even numbers are every third value. It doesn't take much to prove this, so we will do so for rigour. 

There are two main points to note: 

* The sum of two odd numbers is an even number, 
* The sum of an even and odd number is an odd number.

Therefore, if $$F_{n-2}$$ is odd (or even), and $$F_{n-1}$$ is even (or odd), $$F_n$$ will be an odd number. If both $$F_{n-2}$$ and $$F_{n-1}$$ are odd, then $$F_{n}$$ will be even. Representing the series as a pattern of odd's and even's, and noting that the first two numbers are odd and even respectively, we get:

$$\begin{eqnarray} 
ODD + &EVEN = &ODD     \\
	 &EVEN + &ODD = ODD 	 \\
     & &ODD + ODD = EVEN\\
\end{eqnarray}$$

We can see that the fourth item in this series is the same as the first, showing that we do indeed have a cyclical pattern of $$ ODD, ODD, EVEN $$ as observed earlier. Given this, if we can modify the original Fibonacci formula to reference every third value, and initialise on an even number, then we will not need to check every value. 

To start, recall that

$$F_n = F_{n-1} + F_{n-2} $$

From here, we expand out $$ F_{n-1}, F_{n-2} $$, to get

$$\begin{aligned} 
F_n &= F_{n-2} + F_{n-3} + F_{n-3} + F_{n-4}  \\
      &= F_{n-2} + 2F_{n-3} + F_{n-4} \\
      &= F_{n-3} + F_{n-4} + 2F_{n-3} + F_{n-4} \\
      &= 3F_{n-3} + 2F_{n-4} \\
\end{aligned}$$

So far we have the term from 3 steps ago, just need to remove the rest. Expanding one $$F_{n-4}$$, we get:

$$\begin{aligned} 
      F_n &= 3F_{n-3} + F_{n-4} + F_{n-5} + F_{n-6} \\
\end{aligned}$$

and recalling that $$F_n = F_{n-1} + F_{n-2} $$, we subsitute the $$n-4$$ and $$n-5$$ term, ending with:

$$F_n = 4F_{n-3} + F_{n-6} $$

Now that we have a formula for the Fibonacci series that uses every third value, we can initialise with the first even values and now sum all even numbers without needing to check all integers. This is shown in the code snippet below. 

{% highlight python %}

#Initialise these values as the first and second even fibonacci numbers
older_even_term = 2
recent_even_term = 8

limit = 4000000 
answer = older_even_term + recent_even_term
next_even_term = 4*recent_even_term + older_even_term

while next_even_term <= limit:
    answer += next_even_term
    older_even_term = recent_even_term
    recent_even_term = next_even_term
    next_even_term = 4*recent_even_term + older_even_term
    
print(answer)

{% endhighlight %}