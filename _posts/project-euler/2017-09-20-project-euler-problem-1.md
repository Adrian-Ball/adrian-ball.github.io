---
layout: article
title: "Problem 1"
date: 2017-09-20
categories: project-euler
author: adrian-ball
comments: true
image:
  teaser: crayon-numbers.png
  feature: crayon-numbers-banner.png
---

The problem can be found [here](https://projecteuler.net/problem=1). Paraphrased, we want to find the sum of all natural numbers below 1000 that are a multiple of 3 or 5. 

Being the first problem of the site, the logic for solving this problem is fairly straightforward. We want to iterate through all numbers from 1 to 999 and add them to our answer (initialised at 0) if they are divisible by 3 or 5. The only thing to be wary of here is that we don't want to include numbers that are a multiple of 3 and 5 more than once.

Writing this in Python, we could do something like the following: 

{% highlight python %}
n = 1000; 
#Initialise the answer
answer = 0;

for count in range(1,n):
    if count%3 == 0 or count%5 == 0:
        answer = answer + count
        
print('The answer is:',answer)
{% endhighlight %}

While the above is straightforward, it is worth making a few observations here. We only need to iterate across all of the numbers once, keeping the code clean. Also, by only checking each number only once, we won't run into the possibility of including each number more than once. 

While we have an answer to the problem now, looking at the problem from a different angle provides an alternative solution through the use of an arithmetic sum. Putting numerical interest aside, this observation could be beneficial in later problems. Generating an arithmetic sum utility function now allows us the opportunity to have a tested building block towards a solution for a future problem.

To start to build a solution, lets look at adding up all numbers that are a multiple of 3, 

$$ 3 + 6 + 9 + ... + 999 $$

Rewriting this, we get

$$ 3 + (3+1\cdot3) + (3+2\cdot3) + ... + (3+332\cdot3) $$


Now all we need to do is figure out the formula for an arithmetic series.

More generally, suppose that $$s_n$$ is the sum of our series, then we have

$$ s_n = (a + 0 \cdot b) + (a + 1\cdot b) + (a + 2\cdot b) + ... + (a + (n-1)b)$$

where $$a$$ is our initial term, $$b$$ is the constant growth of our terms and $$n$$ is the number of terms in our series (in the 3's example, $$ a = 3 $$, $$ b = 3 $$, $$ n = 333 $$). But we could also rewrite the above as

$$ s_n = (l - 0b) + (l - b) + (l - 2b) + ... + (l - (n-1)b)$$

where $$l$$ is the last term in our original series ($$l = a + (n-1)b$$) and we count down from the largest number instead of up from the smallest number.

Combining the two gives

$$\begin{eqnarray} 
2s_n =& a + (a + b) + (a + 2b) + ... + (a + (n-1)b)      \\
	 +& l + (l - b) + (l - 2b) + ... + (l - (n-1)b) 	 \\
\end{eqnarray}$$

so 

$$\begin{eqnarray} 
2s_n =& a + l + a + l + ... + a + l     \\
	 =& n(a+l) 	 \\
\end{eqnarray}$$

yielding

$$ s_n = \frac{n(a+l)}{2}$$

Now that we have this formula, we just have to plug in the values. We will want to add the sum of the series that were multiple of 3 and 5 and then subtract the sum of numbers that were a multiple of 15 so that they are not included twice. I have done this in the code below, where arithmetic_sum(first, last, num) is the self described function.

{% highlight python %}
#We will want to make use of the arithmetic sum here
from arithmetic_sum import arithmetic_sum

n = 999; 

answer = arithmetic_sum(3,(n//3)*3,n//3)   \
         + arithmetic_sum(5,(n//5)*5,n//5) \
         - arithmetic_sum(15,(n//15)*15,n//15)

print('The answer is:',int(answer))
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


