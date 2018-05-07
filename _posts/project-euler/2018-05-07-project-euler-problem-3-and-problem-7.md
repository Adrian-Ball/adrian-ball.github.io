---
layout: article
title: "Problems 3 & 7"
date: 2018-05-07
categories: project-euler
author: adrian-ball
image:
  teaser: project-euler/problem-3/sieve-teaser.jpg
  feature: project-euler/problem-3/sieve-banner.jpg
---

Given that these two problems are both related to prime numbers and are relatively simple, I describe them both here. In [Problem 3](https://projecteuler.net/problem=3), we are searching for the largest prime factor of a number, while in [Problem 7](https://projecteuler.net/problem=7) we want to know what the $$n^{th}$$ prime is. 

To start, we need a method for generating prime numbers. For these problems, the [Sieve_of_Eratosthenes](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes) was implemented. While there are more efficient prime finding methods, such as the [Sieve_of_Atkin](https://en.wikipedia.org/wiki/Sieve_of_Atkin), the Sieve of Eratosthenes is simple to implement and explain, and is also efficient enough for these problems.

The Sieve of Eratosthenes finds all prime numbers up to a specified value. To do this, we start with an array of length equal to this specified value (note that this can be memory intensive for very large values), and mark each element in the array as being prime. Then starting from 2, each number up to the specified max number that has 2 as a factor is marked as a composite (non-prime) number. This process is repeated with the next number in the array thats labelled as prime, and so on.

We can be efficient in this process by only repeating the above process to the square root of the upper limit, and not to the upper limit itself. Suppose that $$n$$ is the upper limit and that $$n^{0.5} < m < n$$. If $$p*m = q$$, such that $$q < n$$, then $$p < n^{0.5}$$. If $$p \geq n^{0.5}$$, then $$q > n$$, which is not allowed. If $$p$$ is prime, then $$q$$ would already be flagged as a composite number. If $$p$$ is a composite number, then $$q$$ would still be flagged as a composite number! This is due to the [Fundamental_theorem_of_arithmetic](https://en.wikipedia.org/wiki/Fundamental_theorem_of_arithmetic), that states that every number greater than 1 can be represented as the product of prime numbers, which means $$q$$ would have been flagged as a composite number in an earlier iteration of the algorithm.

We can implement this prime finding function in Python as follows:

{% highlight python %}
import numpy as np 
import math

#Sieve of Eratosthenes
#Find all prime numbers up to and including num
def eratosthenes (num):
    #All numbers to check
    sieve_numbers = np.ones(num)
    sieve_numbers[0] = 0 #1 isnt a prime
    counter_limit = math.ceil(num**0.5)+1
    for counter in range(1,counter_limit):
        #If hasnt been identified as prime yet
        if sieve_numbers[counter] == 1:
            next_val = counter+counter+1
            while True:
                if next_val >= num:
                    break
                else:
                    sieve_numbers[next_val] = 0
                    next_val += counter+1
    primes = np.add(np.nonzero(sieve_numbers),1)
    return primes[0]
{% endhighlight %}

Now that we can find prime numbers, we can start to solve these problems. First, for Problem 3, we find all primes less than or equal to the square root of the target. Note again, that we dont need to look at primes greater than the root of the target. From here, the target is divided by each prime, starting with the largest. If the result is an integer (no remainder), then the answer has been found. The Python code to achieve this is: 


{% highlight python %}
import sys
sys.path.append('../utility_functions')
import math
from prime_functions import eratosthenes

#Answer cant be larger than the sqrt of the number
upper_lim = math.ceil(600851475143**0.5)
potential_primes = eratosthenes(upper_lim)
#Count down as we want to find largest prime
for counter in range(len(potential_primes)-1, 0, -1):
    if 600851475143 % potential_primes[counter] == 0:
        answer = potential_primes[counter]
        break

print(answer)
{% endhighlight %}

Finally, in Problem 7, we are looking for the $$n^{th}$$ prime. To keep things simple, the prime generator function is called repetitively, but with a higher number each time. This is done until the number of prime numbers returned is sufficiently large. The following Python code achieves this through the use of a while loop:

{% highlight python %}
import sys
sys.path.append('../utility_functions')
from prime_functions import eratosthenes

nth_prime = 10001
pow = 5
#Being a little lazy here, and just going to call prime function until 
#we have the required amount. 
while True:
    max_num = 2**pow
    primes = eratosthenes(max_num)
    if len(primes) >= nth_prime:
        answer = primes[nth_prime-1]
        break
    else:
        pow += 1

print(answer)
{% endhighlight %}




