# Compound Eye Tools

This library automatically counts and measures the ommatidia of compound eyes from images.

## On Counting Ommatidia

My graduate mentor once challenged me to count the ommatidia of vinegar flies (a.k.a. fruit flies, _Drosophila melanogaster_). I quickly learned that they have two compound eyes, each composed of about 1000 light collecting facets called ommatidia. I learned this as I counted the ommatidia of over 100 flies, culminating in over _90,000_ ommatidia (and [a publication](https://www.sciencedirect.com/science/article/pii/S0042698918300919))!

|[![Image](figs/count_demo.png)](figs/count_demo.png)|
|:--:|
|*Over 90,000 ommatidia from more than 100 flies that I counted by hand. My notes are in white. The geometric arrangement allowed me to count large triangles of ommaitida instead of individuals. Still, a lot of counting.*|

Many have braved this greuling task of counting ommatidia because, among other reasons, compound eyes are the most common image forming structures on Earth. Compound eyes are used by arthropods, which occupy nearly every ecological niche and represent roughly 80% of described animal species (the massive blue slice of the pie of life below). Compound eyes vary substantially in shape and size, having adapted to the gamut of selective pressures. So, to understand how vision works, develops, and evolves, we must study compound eyes. 

|![Image](figs/biodiversity.png)|
|:--:|
|*The Pie of Life: notice the large blue pie representing the arthropods (pronounced "insects")*|

Plus, as opposed to our camera-type eyes, many of the structures limiting the visual performance of compound eyes can be measured in direct external images. The number and size of ommatidia set physical limitations on what a compound eye can see. Each one collects a pixel of the image captured by the retina and sits immediately adjacent to the next ommatidium, separated by screening pigment (see below). The angle separating neighboring ommatidia, called the interommatidial angle, determines the smallest resolvable detail just like the inter-receptor angle of our own camera-type eyes. The size of the ommatidial aperture limits the amount of light available for absorption so that large ommatidia offer greater sensitivity to low light levels. Counting and measuring ommatidia tells us a great deal about the spatial resolution and light sensitivity of many arthropods. 

|![Image](figs/resolution.png)|
|:--:|
|*The camera-type eye can have a higher spatial resolution because one lens serves the whole retina. For a compound eye to acheive the same resolution as our eyes, for instance, it would need to be at least 1 meter in diameter and composed of millions of ommatidia!*|

Wouldn't it be nice if a program could count these ommatidia automatically? Humanity has developed programs that can beat grand champions in chess or generate fake videos of past presidents. If computers can drive cars and recognize faces, shouldn't they be able to detect ommatidia? Introducing the

## Ommatidia Detecting Algorithm (ODA) 

|![Image](figs/oda_demo.png)|
|:--:|
|*General Pipeline of the ODA*|


## Installation
## Simple Recipes
### Individual Image
### Image Stack
### CT Stack (ODA-3D)