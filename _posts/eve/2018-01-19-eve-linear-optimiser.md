---
layout: article
title: "Capitalising on Linear Optimisation!"
date: 2018-01-19
categories: eve-online
author: adrian-ball
comments: true
image:
  teaser: eve/leviathan_over_london_teaser.jpg
  feature: eve/leviathan_over_london_banner.jpg
---

Eve Online is a sandbox game that I have spent some time on. Those interested in it can investigate it furthere [here](https://www.eveonline.com/). One of the things I like to do in this game is to come up with innovative ways to apply interesting math/computer science techniques for some benefit. In this instance I have explored the idea of a linear optimiser for capital ship production (hence the terrible title).

The ultimate goal here is to get a capital class ship to a particular part of the game universe in an efficient manner. The biggest issue is that routes to the space that I reside cannot support capital size craft (for the initiated, I reside in low-class wormhole space). To provide some perspective, a visual scale of the craft in game can be seen [here](http://www.garia.net/g6/eve_chart-4096.jpg), with the capital ships being the 'more obvious' ships in the image.

So, if we cannot take a built ship to where we want it, then the only remaining solution is to move it piece-wise and assemble on site. There are a few choices of how far back in the production chain we go for selecting which 'pieces' we transport to the assembly site. These are shown below in the simplified construction pipeline.

$$ Ore/Asteroids \rightarrow Minerals \rightarrow Ship Parts \rightarrow Ship $$

While this is useful, and provides a solution, there is a problem. The number of trips required for a hauling/logistics ship is extrodinarily large, approximately 30. No thanks! There is however a final option, compressed ore. Suspending belief for a minute (this is a game after all), the volume of minerals produced from compressed ore is larger than the volume of the compressed ore itself!

The challenge now is to figure out which ore we want so that we can build our ship. There are 16 different ore types (that each come in three different grades), and they all refine into different quantities of 8 different minerals. This is where the linear optimiser comes into use!

Linear optimisation is a technique for finding a minimised (or maximised) solution to a model that can be defined by a set of linear relationships. Working backwards, we can convert the ship we want to build into a list of required minerals. The optimiser will then tell us the quantity of each ore needed such that we have the required minerals. The equations, if we were to write them out, would look like

$$ W_1*Ore_1 + W_2*Ore_2 + \cdots + W_{47}*Ore_{47} + W_{48}*Ore_{48} \geq Mineral \ 1 \ requirement$$

where $$W_i$$ is how many of $$Ore_i$$ is required. There would be 8 of these equations in total, one for each mineral. A trivial solution to this would be to set each $$W_i$$ to infinity, however this might be a little expensive and buying all the ore from the market is probably a bad business decision! To counter this, an objective function is needed, something that we are trying to minimise. In this instance I have chosen to minimise the cost of the ore (another option could be to minimise the ore volume) as shown below.

$$minimise(W_1*Ore_{1}cost + \cdots + W_{46}*Ore_{48}cost)$$

After assembling all of this in Google Sheets, building the optimiser with [their API](https://developers.google.com/optimization/lp/add-on), including game related aspects (which I won't bother to explain here), and even using some Eve Online API to get market data out of game, a working model is developed which can be seen [here](https://docs.google.com/spreadsheets/d/1w0j7Dnh9hmOws4ZaxpF6-p-i2nkxnr-pYzf7Ib4BMws/edit?usp=drive_web). It is at this point I should thank Gram, an in-game accomplice, for assistance with the interface to make it more user friendly.

Testing the optimiser, the compressed ore required to build a capital ship will fit into the cargo bay of a single hauling/logistics ship! This is a significant improvement on the estimated 30 trips from earlier. Also, given that we optimised for cost and arrived at a solution that requires the minimum number of trips, there is no need to repeat the process with an objective function that minimises the total volume of the compressed ore (risking a potentially significant cost increase). An added bonus now that this is up and running is that it can be (and has been) used for cost minimisation in other in-game industrial ventures.


<sup>
EVE Online and the EVE logo are the registered trademarks of CCP hf. All rights are reserved worldwide. All other trademarks are the property of their respective owners. EVE Online, the EVE logo, EVE and all associated logos and designs are the intellectual property of CCP hf. All artwork, screenshots, characters, vehicles, storylines, world facts or other recognizable features of the intellectual property relating to these trademarks are likewise the intellectual property of CCP hf.
</sup>

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