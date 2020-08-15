---
layout: article
title: "Problem 58"
date: 2020-08-15
categories: project-euler
author: adrian-ball
comments: true
image:
  teaser: project-euler/problem-58/ulam-teaser.png
  feature: project-euler/problem-58/ulam-banner.png
---

In [this](https://projecteuler.net/problem=58) problem, we are looking at the diagonal values of an counter-clockwise spiral (example shown below), and counting how many of them are prime. In the shown example, eight of the thirteen numbers are prime, i.e., 62%, and we are tasked with finding what the side length of the square spiral is when the ratio of primes first drops below 10%.

<p align="center">
  <img width="200" height="200" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Ulam_spiral_howto_all_numbers.svg/200px-Ulam_spiral_howto_all_numbers.svg.png">
</p>

For some history, this type of spiral when the primes are singled out is known as an Ulam spiral (Originating from [this](https://www.semanticscholar.org/paper/A-VISUAL-DISPLAY-OF-SOME-PROPERTIES-OF-THE-OF-Stein-Ulam/76879c5908f7d8e289642b0ab8943529eb39e1d9) paper). Ulam thought that there was a non-random pattern that appeared in such a spiral. An image of a 399x399 spiral is shown below. 

<p align="center">
  <img width="400" height="400" src="https://mathworld.wolfram.com/images/gifs/primesp.jpg">
</p>

There is also an argument to be made that perhaps the Ulam spiral should be known as a Clarke spiral too. Arthur C. Clarke described the spiral in his novel *The City and the Stars* (1956):

> "Jeserac sat motionless within a whirlpool of numbers. The first thousand primes.... Jeserac was no mathematician, though sometimes he liked to believe he was. All he could do was to search among the infinite array of primes for special relationships and rules which more talented men might incorporate in general laws. He could find how numbers behaved, but he could not explain why. It was his pleasure to hack his way through the arithmetical jungle, and sometimes he discovered wonders that more skillful explorers had missed. He set up the matrix of all possible integers, and started his computer stringing the primes across its surface as beads might be arranged at the intersections of a mesh."

With all that said, let's shift gears to solving the problem!

Conceptually, solving the problem is not too difficult. We just need to check the ratio of primes to non-primes in the current square spiral. If the ratio is still greater than 10%, add another ring to the spiral, then repeat this process until the ratio criteria is met. To do this, we will start with the given values from the initial 7*7 spiral. This gives the following:

{% highlight python %}

#Starting from the given example with sides of 7
numerator = 8.
denominator = 13
sq_size = 7

{% endhighlight %}

From here, each time we add a ring to the spiral (increasing the square side length by 2), we can check the corners to see if they are prime. If so, we can add them to the numerator for our ratio, and add 4 to the denominator. The value for each corner of the square can also be determined given the size of the square. These values are: 

$$
\begin{aligned}
  & n^2 \\
  & n^2 -1*(n+1)\\
  & n^2 -2*(n+1)\\
  & n^2 -3*(n+1)\\
\end{aligned}
$$

where $$n$$ is the side length of the square. Note that we dont have to check the $n^2$ case, as it is obviously not a prime. 

Folding all this into a loop, we have the following:

{% highlight python %}

while(True):
    #Increase the size of the square
    sq_size += 2
    
    #Check 3 of the 4 corners to see if they are prime.
    for i in range(1,4):
        corner = sq_size**2 - i*(sq_size-1)
        if is_prime(corner):
            numerator += 1
            
    #Add each corner to the denominator 
    denominator +=4
    
    if numerator/denominator < 0.1:
        print(sq_size)
        break

{% endhighlight %}

In the above code snippet, we have a function ```isprime()```. This function (obviously) checks whether the provided number is prime, and is where some of the hidden work occurs. In the solving of earlier questions, the [Sieve of Eratosthenes](https://mathworld.wolfram.com/SieveofEratosthenes.html) was introduced as a method for the generation of primes. While cross checking against this list is a viable option to generating a solution, it is a somewhat time intensive process. A much faster process is the *Rabin-Miller Primality Test* (two standard links [here (Wiki)](https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test) and [here (Wolfram)](https://mathworld.wolfram.com/Rabin-MillerStrongPseudoprimeTest.html)). In short, this test returns a 'soft' answer on whether the given number is prime. If the returned answer is that the number is a composite (non-prime), then this is definitely the case. However, if the returned answer is that the number is a prime, then there is potentially some non-zero chance that the number is actually a composite. 

I won't cover the specifics of the method here (the previous links provide a good overview), but the test loosely revolves around the computation of some number $$n$$ with a random number $$a < n$$. The interesting thing to note is that the test can be repeated for multiple values of $$a$$, and the more tests that pass, the more certainty there is that the number $$n$$ is in fact prime. For problems such as this one, the Miller-Rabin Primality Test is good, as we have to test numbers multiple times, and the computational complexity of the algorithm is $$O(k*log^3n)$$, where $$k$$ is the number of tests performed. 

There is a deterministic version of this test (though it is reliant on the unsolved [Riemann Hypothesis](https://en.wikipedia.org/wiki/Generalized_Riemann_hypothesis#Extended_Riemann_hypothesis_(ERH))). The evaluation of the deterministic tests is documented [here](https://miller-rabin.appspot.com/) for suitable bases and upper-bound limits. For this problem however, the non-deterministic function was sufficient for yielding the correct answer to the current Project Euler question. Finally, I will round out this article with a python implementation of this test: 

{% highlight python %}

#Miller-Rabin Primality Test (non-deterministic version)
import random
def is_prime(n):

    #Write n as 2**s * d + 1
    #           2**twos_exponent * twos_coeff + 1
    #or   n-1 = 2**twos_exponent * twos_coeff 
    twos_coeff = n-1
    twos_exponent = 0
    while twos_coeff % 2 == 0:
        twos_coeff >>= 1 #Bitshift
        twos_exponent+=1

    #Check for whether number is composite 
    #(Failing only suggests at it being prime)
    def trial_composite(a):
        if pow(a, twos_coeff, n) == 1:
            return False
        for i in range(twos_exponent):
            if pow(a, 2**i * twos_coeff, n) == n-1:
                return False
        return True

    #Run a bunch of tests, if they all pass then we guess that n is prime
    #If one test fails, then we know that the number is a composite
    for _ in range(10):
        a = random.randrange(2, n)
        if trial_composite(a):
            return False
    return True

{% endhighlight %}

{% if page.comments %}
<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://https-adrian-ball-github-io.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
{% endif %} 