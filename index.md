---
layout: archive
permalink: /
title: "Adrian Ball's Page"
image:
  feature: studio_panorama_banner.jpg 
---

<p>
Hello and welcome to my personal site. I have generated this website so that I have a place to present myself along with interesting ideas and concepts. My hopes is that this site can be used as a platform to present ideas, as a motivational basis to continually develop presentation skills with a growing portfolio, and to show more than just a curriculum vitae.
</p>

<p>
Please feel free to navigate the site, read any content of interest to yourself, and to reach out through Disqus or email - adrian.k.ball85@gmail.com
</p>

<h2> Latest Posts </h2>

<div class="tiles">
{% for post in site.posts limit:8 %}
  {% include post-grid.html %}
{% endfor %}
</div>

