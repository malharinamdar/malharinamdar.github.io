---
layout: post
title: "Anylogic and Slack"
description: "Sending messages to Slack when simulation is done"
author: Sebastian Daza
comments: true
date: 2050-05-01
tags: 
  - anylogic
  - ABM
---

When running a parameter variation experiment, that is, simulating over several iterations and replicates using parallelization, we usually need to collect a huge amount of data and have them in a format that then we can process using Python or R.

The best way to do this in Anylogic would be using a database and then export, read, or connect to the database to process simulation results (although, see the section **update** below). We can do that easily in Anylogic. Every time an experiment finishes, we can export the data (from a database) to an Excel file manually.

The issue with Excel files is, on the one hand, they are Excel files, and on the other, they are not suitable for big data (more than 1 million rows). We can create a function to save all the simulation tables into an Excel file as our experiment finishes. However, we will still have the limit-of-rows limitation (check the Anylogic file linked below for a function to create Excel files from a database).

Here I follow a different approach by exporting an Anylogic database table to a CSV file within Java. The general setup using Anylogic PLE 8.6 would be:

- Create the databases you need for your experiment, and be sure you add the columns iteration and replicate.
- Create a function to save the data of your simulation runs (e.g., agent's status, age, etc.)
- Define a parameter variation experiment.
- Define a variable to specify where to save the data (i.e., path).
- Write code in the experiment Java Actions section so that to save data every time you run an experiment.
- Import functions in the advanced Java section of the experiment.

## Databases

I define two tables (`data1` and `data2`). Each agent saves its data at a given rate. After 5 years, the simulation will finish.

{% highlight java %}
insertInto(data1)
    .columns(data1.iteration, data1.replicate, data1.id, data1.dtime, data1.drandom)
    .values(main.v_iteration, main.v_replicate, this.getIndex(), time(), normal(1.0))
    .execute();

insertInto(data2)
    .columns(data2.iteration, data2.replicate, data2.id, data2.dtime, data2.drandom)
    .values(main.v_iteration, main.v_replicate, this.getIndex(), time(), normal(0.3))
    .execute();

if (time() > 5) {
    finishSimulation();
}
{% endhighlight %}

The tables include a column with the experiment iteration and replicate, in addition to the agent's index, time, and a random value from a normal distribution.

## From DB to CSV

The key function to export the data to a CSV file is `f_SQLToCSV`. It uses two arguments, a SQL query (`query`) and the path to an output file (`filename`). For instance, we can write:

{% highlight java %}
f_SQLToCSV("select * from data1", "output/data1")
{% endhighlight %}

You can use any query for your data, giving you a lot of flexibility on what to export to a CSV file. The `f_SQLToCSV` method is:

{% highlight java %}
ResultSet rs = selectResultSet(query);
try {
    File file = new File(filename + ".csv");
    file.getParentFile().mkdirs();
    FileWriter fw = new FileWriter(file);
    int cols = rs.getMetaData().getColumnCount();
    for(int i = 1; i <= cols; i ++){
        fw.append(rs.getMetaData().getColumnLabel(i).toLowerCase());
        if(i < cols) fw.append(';');
        else fw.append('\n');
    }
    while (rs.next()) {
        for(int i = 1; i <= cols; i ++){
            fw.append(rs.getString(i));
            if(i < cols) fw.append(';');
        }
        fw.append('\n');
     }
     fw.flush();
     fw.close();
} catch (Exception e) {
    getEngine().pause();
    traceln("--> An Exception happened, continue? ...");
    e.printStackTrace();
}
{% endhighlight %}

The next step would be to create an experiment and complete the Java actions accordingly. First, we clear our tables.

{% highlight java %}
// inital experiment setup
deleteFrom(data1).execute();
deleteFrom(data2).execute();
{% endhighlight %}

Then, we collect information on the iteration and replicate of the simulation run:

{% highlight java %}
// before simulation run
root.v_iteration = getCurrentIteration();
root.v_replicate = getCurrentReplication();
{% endhighlight %}

Finally, at the end of the experiment, we save the data and clear the tables again:

{% highlight java %}
// after experiment
f_exportTables(v_path);
deleteFrom(data1).execute();
deleteFrom(data1).execute();
{% endhighlight %}

The method `f_exportTables` is just a function that goes through each table and export them to a CSV file. `v_tables` is string array with the name of the tables I want to export `{"data1", "data2"}`:

{% highlight java %}
ArrayList<String> tables = new ArrayList<String>();
for(String tab : v_tables) {
   tables.add(tab);
}
for (String t : tables) {
    f_SQLToCSV("select * from " + t, path + t);
}
{% endhighlight %}

Remember to import some functions in the `imports section`:

{% highlight java %}
// imports section
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.text.*;
{% endhighlight %}

From there, we can create additional functions to select the tables to be exported. For more details, download the [Anylogic File here](/assets/files/DBToCSV.zip).





