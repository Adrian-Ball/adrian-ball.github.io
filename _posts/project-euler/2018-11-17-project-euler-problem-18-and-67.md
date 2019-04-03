---
layout: article
title: "Problem 18 & 67"
date: 2018-11-17
categories: project-euler
author: adrian-ball
comments: true
image:
  teaser: project-euler/problem-18-67/pascal_triangle_teaser.png
  feature: project-euler/problem-18-67/pascal_triangle_banner.png
---

As [Problem 18](https://projecteuler.net/problem=18) and [Problem 67](https://projecteuler.net/problem=67) are the same (Problem 67 is a larger version of Problem 18), I will cover the solution to both in this post. Both of these problems have a triangle of numbers, with the objective of finding the maximum sum from the apex to base of the triangle. When going from one row in the triangle to the next, an adjacent number must be chosen.

From the description of Problem 67, we are given the following note: *"It is not possible to try every route to solve this problem, as there are $$2^{99}$$ altogether! If you could check one trillion (1012) routes every second it would take over twenty billion years to check them all."* This makes it mandatory for an efficient algorithm to be found (of course we want to find an efficient method that generates a solution anyway).

Given that the note describes the difficulty with a brute force solution, we move to immediately look at the efficient method for solving these problems. Given an arbitrary row, $$n$$, it is easy to see the max sum from unit-length paths that start from row $$n-1$$. For each of the ($$n$$) elements of row $$n$$, we look at its two parents and add the largest to the element (Note that the end elements only have one parent). We can then look through the row of modified elements for the largest value.

The method desribed above can be repeated, resulting in an 'induction like' solution. Given the first row, we find the maximum value for a path to each element of the second row. We then replace the second row with the max-sum paths to each element of the row. By repeating this process for as many rows as there are, the final row will contain a list of max-sum paths for each of the elements. Taking the maximum of these elements will provide the greatest possible sum from apex to base of the number-triangle.

The python code to solve this problem is shown below, and should be fairly straightforward to follow. Note that the previous row defined as the current row once the necessary operations on the current row have been performed so that we can iterate down the rows of the triangle.

{% highlight python %}
file_path = 'p067_triangle.txt'
with open(file_path) as file:
    #Treat the first line as a special case, no predecessors
    prev_row = list(map(int, str.split(file.readline())))
    for line in file:
            curr_row = list(map(int, str.split(line)))
            curr_row[0] += prev_row[0]
            curr_row[-1] += prev_row[-1]
            #Determine the larger of the two parent elements
            #and then add to current element. 
            for element in range(1, len(curr_row)-1):
                max_prev = max(prev_row[element-1],prev_row[element])
                curr_row[element] += int(max_prev)
            prev_row = curr_row
            
answer = max(prev_row)
print(answer)
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