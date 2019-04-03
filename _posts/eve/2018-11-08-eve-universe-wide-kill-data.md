---
layout: article
title: "Temporal Visualisation of Kill Data in the Eve Universe"
date: 2018-08-11
categories: eve-online
author: adrian-ball
comments: true
image:
  teaser: eve/temporal_visualisation_kill_data/daredevil_teaser.jpg
  feature: eve/temporal_visualisation_kill_data/daredevil_banner.jpg
---

This article discusses a small project I undertook to generate a gif showing temporal NPC (non-playable character) ship kill data across the Eve Online universe, New Eden. Killing NPC's is something that can be done in game when trying to complete a mission or acquire some protected. The most important thing (to the player) is that killing NPC's often generates in game money. The project was of interest to me as it had multiple stages, with each stage improving my current skills and often providing the chance for me to learn something new. I also enjoy exploring what can be done with the game beyond directly playing it. To explain how the project came together, I have broken this article into several sections, each discussing a different aspect of the project.

<h5>Stage 1: Data Acquisition</h5> 

CCP have an API that allows for interested parties to acquire a variety of information about the Eve universe, and is typically used by third party developers that build apps to augment their gaming experience (such as skill planners, market analysers, etc). For this project, we are interested in data related to the destruction of ships in game. This particular function can be found 
<a href="https://esi.evetech.net/ui#/Universe/get_universe_system_kills"> here </a>, and a snippet of the data from a call can be seen below. We can see that three types of kills are recorded for each system. The only 'gotcha' here, is that systems that have no recorded kill data will not be on this list. This will be more relevant later. 

{% highlight python %}
[
  {
    "npc_kills": 1660,
    "pod_kills": 0,
    "ship_kills": 0,
    "system_id": 30004663
  },
  {
    "npc_kills": 77,
    "pod_kills": 0,
    "ship_kills": 0,
    "system_id": 30004995
  },
  {
    "npc_kills": 161,
    "pod_kills": 0,
    "ship_kills": 1,
    "system_id": 30004972
  }
]
{% endhighlight %}

Given that each API call provides data for the past hour, and that a call will nominally need to be made once an hour (data is cached for up to one hour), some form of timed script is required. To do this, I set up an AWS Lambda function. Such a function, from the <a href="https://aws.amazon.com/lambda/features/"> AWS website</a>, "is a serverless compute service that runs your code in response to events and automatically manages the underlying compute resources for you". Setting up a timer as an event with a lambda function means that I can regularly acquire data, and given that I am running a cloud function (versus something at home on my local computer), I can expect to consistently obtain data. 

The code written for the AWS lambda function is shown below. It performs the API call as described above, but also extracts some date and time information from the header of the data. This information is useful as it tells us the time period in which the kill data was obtained. From here, the data is then written into a csv file, which is saved in S3, Amazon's cloud storage system.

{% highlight python %}
import boto3
from botocore.vendored import requests
import os

def lambda_handler(event, context):
    
    s3 = boto3.resource('s3')
    
    url_response = requests.get('https://esi.tech.ccp.is/latest/universe/system_kills/?datasource=tranquility')
    last_modified = url_response.headers['last-modified'] 
    #edited_stamp = last_modified[5:len(last_modified)] + ', ' + last_modified[0:3]
    split_last_modified = last_modified.split()
    #year, month, day, time, weekday(dont want)
    edited_stamp = split_last_modified[3] + ' ' + split_last_modified[2] + ' ' + split_last_modified[1] + ' ' + split_last_modified[4] + ' ' + split_last_modified[0][0:3]
    file_to_check = edited_stamp + '.csv'
        
    data = url_response.json()
    
    with open('/tmp/' + edited_stamp + '.csv', 'w') as file:
    
        file.write('npc_kills, pod_kills, ship_kills, system_id')
        file.write('\n')
        for i in range(0, len(data)):
            current_system = data[i]
            npc_kills = current_system['npc_kills']
            pod_kills = current_system['pod_kills']
            ship_kills = current_system['ship_kills']
            system_id = current_system['system_id']
            
            file.write(str(npc_kills) + ', ' + str(pod_kills) + ', ' + str(ship_kills) + ', ' + str(system_id))
            file.write('\n')
        
    s3.Object('eve-online-system-kill-data', edited_stamp + '.csv').put(Body=open('/tmp/' + edited_stamp + '.csv', 'rb'))
{% endhighlight %}

Finally, these csv files can be downloaded through a script on my local machine. In order for this to work, some permission credentials needed to be set up. Instructions on how to do that can be seen <a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-config-files.html">here</a>. The code that copies the csv files from AWS S3 and stores them on my local machine is below.

{% highlight python %}
import boto3
import botocore
import pandas
import os

s3_resource = boto3.resource('s3')
bucket_name = 'eve-online-system-kill-data'
bucket = s3_resource.Bucket(bucket_name)

for key in bucket.objects.all():
    current_file = '../../data/eve_data/downloaded_system_kill_data/' + key.key
    if not os.path.exists(current_file):
        #file doesnt exist, so download
        print(current_file)
        s3_resource.Bucket(bucket_name).download_file(key.key, current_file)
{% endhighlight %}

<h5>Stage 2: Building a Database</h5>

Now that I have the data on my local machine, I need to reorganise it in such a way that I can perform some analysis. To accomplish this, I built a script that takes the kill data from the csv files and then collates it in a SQL database. 

Before this can be done though, more data about the Eve universe needs to be acquired. As I eluded to earlier, the kill data that I have collected through API calls does not included systems where no kills have happened. To be clear in my database that there were 0 kills in a system, I would like to this explicitly stated in the database, rather than having that system being absent at the corresponding timestamp. Doing this means I need to check the systems I do have data for against a list of all possible systems. From here, the missing systems can be entered into the database with kill values of 0.

Obtaining a list of all systems in the Eve universe could be done through an API call in a similar manner to how I obtained the kill data. However, to explore a different avenue for the purpose of trying new things, I used the <a href="https://developers.eveonline.com/resource/resources">Eve Static Data Export (SDE)</a> which provides information about all constant data in the game. Specifically, I used an SQL version of the SDE, which can be found through a link on the SDE page. 

To start the import of data from csv files into the database, I first (after overhead code to initialise the databases) need to identify which files have already been copied in, so that only new csv files are processed. This is done through the following code snippet. Here, I form 2 sets of filenames; all csv files in the directory, and all files that currently exist in the database. The difference between these two sets are the remaining files to be processed. 

{% highlight python %}
filenames = {filename for filename in os.listdir(directory)}
kd_cursor.execute('SELECT DISTINCT file_name FROM system_kill_data;')
files_already_processed = {row[0] for row in kd_cursor.fetchall()}
unprocessed_files = filenames - files_already_processed
{%endhighlight%}

Once a list of all unprocessed files has been generated, it is just a simple matter of iterating through these files and adding their content to the database. The only thing to be wary of here is that we include the systems that have not had any kills. To do this, a list of present systems is generated as the csv is processed, and then compared to a list of all systems. Again, taking the set difference of these two lists identifies the systems that have had 0 kills. Finally, the new data to be stored in the database is added to a variable that is saved to the database once all files have been processed. This is done so that only one write to the database is required. The code that accomplishes this is shown in the snippet below.

{% highlight python %}
data_to_add = []
#Iterate over all files
for filename in tqdm(unprocessed_files):

    split_name = filename.split()
    timestamp = (split_name[0] + month_dict[split_name[1]] + split_name[2] +  
                split_name[3][0:2] + split_name[3][3:5] + split_name[3][6:9])

    with open(directory + filename, 'r') as current_data:
        next(current_data, None) #Skip header
        reader = csv.reader(current_data)
        systems_present_in_file = set()
        for row in reader:
            entry_data = [filename, timestamp, split_name[0], month_dict[split_name[1]], split_name[2],
                            split_name[3], split_name[4][0:-5], row[3], row[2], row[1], row[0]]
            data_to_add.append(entry_data)
            systems_present_in_file.add(int(row[3]))

    missing_systems = set_of_systems - systems_present_in_file  

    for curr_system in missing_systems:

        entry_data = [filename, timestamp, split_name[0], month_dict[split_name[1]], split_name[2],
                    split_name[3], split_name[4][0:-5], curr_system, 0, 0, 0]
        data_to_add.append(entry_data)

kd_cursor.executemany('INSERT INTO system_kill_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', data_to_add)
print('There were ' + str(len(unprocessed_files)) + ' files added to the database')
{%endhighlight%}

<h5>Stage 3: Choosing a Data Time Range</h5>  

To start the process of generating the images for my gif, I first need to set up access to the relevant databases. From here, I can have a look at how much data and which data I want to analyse for the gif.

{% highlight python %}
import os
import pandas as pd
import sqlite3 
import imageio
import matplotlib

conn_kill_data = sqlite3.connect('../../data/eve_data/eve_db.db')
conn_eve_sde = sqlite3.connect('../../data/eve_data/eve_sde/sqlite-latest.sqlite')
{% endhighlight %}

My objective was to find a period of time that was dense with data. The final data range that I chose had the expected number of entries at 96 (4 days with an entry each hour), and the query to do this can be seen in the snippet below. For some of the data, the time between each data set that I acquire  from the API calls is greater than one hour, which is longer than I expected. Investigating the cause of this will be a task for when this project is complete. 

{% highlight python %}
query_datacount = 'SELECT DISTINCT(timestamp) FROM system_kill_data WHERE month=4 AND day>26;'
timestamp = pd.read_sql(query_datacount,conn_kill_data)
print(len(timestamp))
{% endhighlight %}

Now that I know the time of the data that I am interested in, I can set up database queries to acquire the relevant information. First, I get the kill data from my database. Following this is another query to the SDE database so that the coordinates of each solar system can be obtained for when it is time to plot. The `system_id` requirement of the queries is to ensure that I do not access the 'wormhole systems' of the universe, which do not form part of the static map. Finally, the data obtained from each query are merged together into a single table.

{% highlight python %}
query_get_data = 'SELECT timestamp, time, system_id, ship_kills, npc_kills, pod_kills' \
                + ' FROM system_kill_data' \
                + ' WHERE month=4 AND day>28 AND system_id < 31000000;'
kill_data = pd.read_sql(query_get_data,conn_kill_data)

query_solar_system_coords = 'SELECT x,y,z,solarSystemID FROM mapSolarSystems WHERE solarSystemID < 31000000;'
data_solar_system_coords = pd.read_sql(query_solar_system_coords,conn_eve_sde)
combined_data = data_solar_system_coords.set_index('solarSystemID').join(kill_data.set_index('system_id'))
{% endhighlight %}

Now that all of the data is available, the individual image frames can be generated. One thing to note here is that the spread of NPC kills is not linearly spread between 0 and the maximum. To avoid the few systems with a relatively large NPC kill count saturating out the other systems, the colour and size of all systems were scaled non-linearly. The final scaling factors that I settled with can be seen in the code below. 

{% highlight python %}
max_npc_kill = combined_data['npc_kills'].max()
unique_timestamps = sorted(combined_data.timestamp.unique())
#plotfont
font = {'family': 'serif',
        'color':  'gray',
        'weight': 'normal',
        'size': 14,
        }
img_save_dir = '../../data/eve_data/eve_system_kills_temporal/img/png/'
mov_save_dir = '../../data/eve_data/eve_system_kills_temporal/img/gif/'

for curr_ts in unique_timestamps:
    data_particular_timestep = new_data.loc[new_data['timestamp'] == curr_ts]
    data_particular_timestep.loc[:,'colour'] = (data_particular_timestep.loc[:,'npc_kills']/max_npc_kill)**0.25
    data_particular_timestep.loc[:,'plot_size'] = (data_particular_timestep.loc[:,'npc_kills']/max_npc_kill)**0.5
   
    plt = data_particular_timestep.plot.scatter(x='x',y='z', \
                                                s=25*data_particular_timestep['plot_size'], \
                                                c=data_particular_timestep['colour'], \
                                                cmap='summer', \
                                                figsize=(16,12))
    
    plt.set_facecolor('black')
    plt.axis('tight')
    plt.axes.get_xaxis().set_visible(False)
    plt.axes.get_yaxis().set_visible(False)
    
    #Hide the colourbar
    f = matplotlib.pyplot.gcf()
    cax = f.get_axes()[1]
    cax.remove()

    ts = data_particular_timestep['timestamp'].as_matrix()
    ts = ts[0]
    
    plt.text(0.01, 0.01, 'Timestamp: ' + str(ts), fontdict = font, transform=plt.transAxes)
    
    matplotlib.pyplot.savefig(img_save_dir + 'img_' + str(curr_ts) + '.png')
    if curr_ts != unique_timestamps[-1]:
        matplotlib.pyplot.close()
{% endhighlight %}

<h5>Stage 4: Building the Image</h5>  

Now that all of the images have been generated, they can now be collated into a single gif. The code for doing this can be seen below, with the final image below that. In it, you can see a visual representation of the number of NPC killed in each system over time with the changing size and colour of each system. For the timestamp, its representation is yyyy-mm-dd-hh-mm-ss.

{% highlight python%}
universe_images = []
for file_name in os.listdir(img_save_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(img_save_dir,file_name)
        universe_images.append(imageio.imread(file_path))
        
imageio.mimsave(mov_save_dir + 'movie.gif', universe_images, duration=0.7)
{% endhighlight %}

![The final gif!](../../images/eve/movie.gif "The final gif!"){: .center-image}

Looking at the image, it is no surprise to see a constant presence of kills in the centre of the universe. This is some of the 'safer' space that exists, and as a consequence, players are more likely to perform 'missions'  in this space. Missions are tasks that often require the player to eliminate waves of NPCs. The outskirts of the universe are more lawless, and is space that players can fight over and conquer. In this type of space, we can see that the number of NPC kills changes a lot more over time. This is likely due to be to players working together in a timezone, offering more relative safety to one another, or to NPC killing being interrupted as players find ways to avoid losses to enemies. Finally, the bottom left of the universe is, at the moment, occupied by one of the most organised groups in the game. We can see that their organisation, high player count, and deployed infrastructure result in many players profiting off the death of NPCs.

<h5>Conclusion</h5>  

This page outlays the design and implementation of a personal project. The objective of this project was to generate a temporal image of the Eve universe, showing the changing kill data over time. I managed to complete this objective and acquire some new skills along the way. Some of the more significant skills, in no particular order, include AWS Lambda functions, jupyter notebooks, pandas, and understanding how to properly set up python environments. Not every obstacle or new skill was discussed in an attempt to keep the article from being (too) bloated :).

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