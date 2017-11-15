---
layout: article
title: "Problem 85"
date: 2017-11-16
categories: project-euler
author: adrian-ball
image:
  teaser: project-euler/problem-85/wooden-toys-original.jpg
  feature: project-euler/problem-85/wooden-toys-banner.jpg
---

In [this](https://projecteuler.net/problem=85) problem we are interested in counting the number of sub-rectangles in a rectangular grid. Specifically, we are interested in the area of a grid that yields the closest possible sum of sub-rectangles to two million. I know this is a random problem to write up after Problem 1, however I had recently re-solved this problem on the train by accident and the solution also makes use of arithmetic series, so we can exploit that straight away!

The figure below shows an example for a rectangular grid with 2 rows and 3 columns. For a grid of this shape, there are 18 sub-rectangles.

![Image from projecteuler.net](https://projecteuler.net/project/images/p085.gif "Image from projecteuler.net"){: .center-image}

The challenge now is to see if we can find an equation that provides us the number of sub-rectangles in a rectangular grid of arbitrary size. We will assume that such a grid has $$m$$ rows and $$n$$ columns. To start, we will try to calculate the number of sub-rectangles of particular sizes and see if we can observe a pattern.

For a $$1\times1$$ rectangle, there are obviously $$mn$$ potential options. For a $$1\times2$$ sub-rectangle, on one row, there are now $$n-1$$ options for where that sub-rectangle can be placed. Doing this for all $$m$$ rows means that there are $$m(n-1)$$ sub-rectangles of size $$n-1$$. Looking at sub-rectangels of size $$1\times1$$ to $$1\times n$$, we get the following:

$$\begin{eqnarray} 
1\times1 =& m\times n  \\
1\times2 =& m\times (n-1)  \\
1\times3 =& m\times (n-2)  \\
&\vdots \\
1\times n =& m\times 1  \\
\end{eqnarray}$$

Does the $$n$$ column above look familiar? It should, as it is an arithmetic sum! We would also see a similar sum for the $$m$$  values if we had grown the sub-rectangles vertically rather than horizontally. To try and be transparent on how the final formula is generated, lets look at sub-rectangles with a height of 2. We would get the following:

$$\begin{eqnarray} 
2\times1 =& (m-1)\times n  \\
2\times2 =& (m-1)\times (n-1)  \\
2\times3 =& (m-1)\times (n-2)  \\
&\vdots \\
2\times n =& (m-1)\times 1  \\
\end{eqnarray}$$

Completing this process would give us the number of sub-rectangles in the original $$m \times n$$ rectangle, and we can see that it is the product of two arithmetic sums. Borrowing from the work [presented here]({% post_url 2017-09-20-project-euler-problem-1 %}) for Problem 1, we get

$$\begin{eqnarray}  
S =& \frac{m(m+1)}{2} \frac{n(n+1)}{2} \\
  =& \frac{m(m+1)n(n+1)}{4}
\end{eqnarray}$$

where $$S$$ is the sum of all sub-rectangles in a rectangular grid of $$m$$ rows and $$n$$ columns.

From here, we just have to evaluate this equation for rectangular grids of different sizes until we find the $$S$$ closest to two million, which in turn will give the answer for this problem. I have done this in Python with two for-loops, one for each dimension of the rectangle. The limit for one of the dimensions is two million, which is probably a little conservative as we expect an $$S$$ close to two million. We can however set the limit for the second dimension lower, and choose a limit such that when fixing the length of one side of the region, the max area is approximately two million. Finally, we just have to check for each rectangle to see if we are getting closer to the final solution. The Python code for this can be seen below:

{% highlight python %}
#We will want to make use of the arithmetic sum here
from arithmetic_sum import arithmetic_sum
import math

answer = 0
distance_to_2mil = 2000000

limit_i = math.ceil(math.sqrt(2000000))
for i in range(1,limit_i):
    limit_j = round(2000000/i)
    for j in range(1,limit_j):
        term_1 = arithmetic_sum(1,i,i)
        term_2 = arithmetic_sum(1,j,j)
        num_rectangles = term_1 * term_2
        if abs(num_rectangles - 2000000) < distance_to_2mil:
            distance_to_2mil = abs(num_rectangles - 2000000)
            answer = i*j
        
print('The answer is:',answer)
{% endhighlight %}




