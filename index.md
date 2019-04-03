---
layout: archive
permalink: /
title: "Personal Site: Adrian Ball"
image:
  feature: studio_panorama_banner.jpg 
---

<p>
Hello and welcome to my personal site. I have built this site so that I can have a place to present myself along with interesting ideas and concepts. The hope is that the site will be useful as a portfolio in a sense, and show more than just a curriculum vitae. I have also never built a website before, so I am gaining new skills as this site develops. 
</p>

<h2> Latest Posts </h2>

<div class="tiles">
{% for post in site.posts %}
  {% include post-grid.html %}
{% endfor %}
</div><!-- /.tiles -->