# weather_fields

## Using ODE/SDE based models to solve inverse problems of increasing resolution of weather fields
 
### Description
	Recently, diffusion models have been the most successful models in terms of generation quality. One view of diffusion models is that they define a stochastic differential equation (SDE) that gradually translates data into noise, which is then time-reversed and numerically solved to generate objects.  
In addition to diffusion models working initially in the paradigm of data generation from noise, a whole family of models (Flow matching, Stochastic interpolants) of comparable quality has appeared, working on the basis of ordinary (ODE) or stochastic (SDE) differential equations and allowing their application to arbitrary pairwise problems (the pair noise <-> data is also suitable). Thus, they can be used to solve inverse problems of an arbitrary kind, i.e., problems in which one needs to restore a corrupted object in some way. These include, for example, resolution enhancement, docking, deblurring, and coloring.

The project proposes to learn about the described family of models, implement them in practice, and apply them to the inverse problem of resolution enhancement of weather forecast field calculations for a given region. Data for test calculations will be provided.
 
### Materials
* https://arxiv.org/abs/2210.02747
* https://arxiv.org/abs/2302.00482
* https://arxiv.org/abs/2307.03672
* https://arxiv.org/abs/2209.15571v3
