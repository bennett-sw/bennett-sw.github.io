# Liu and Yang (2025) Spatial Alignment Model Documentation
## Layout of this documentation repository
- A conceptual overview of the model [here](overview.md)
- A jupyter notebook which walks through the actual code [here](walkthrough.ipynb)
- A reference manual for all commands that are used in the code [here](command_references.md)

## To do (in order of execution)
1. <span style="text-decoration: line-through;">Finish writing out summary of code by chunk </span>
4. Clean up the current .py file into a new one, and complete the following: 
    -  <span style="text-decoration: line-through;">Stop hard-coding paths </span> 
    -  <span style="text-decoration: line-through;">Improve naming conventions, i.e. standardize anchor vs. reference, cut out 'v_2', etc. </span> 
    - <span style="text-decoration: line-through;">Remove extraneous comments</span> 
    - <span style="text-decoration: line-through;">Re-order the code so that it's more intuitive</span>
    - <span style="text-decoration: line-through;">Try and improve the speed of computation --- can we vectorize, or fix the data type (so no sympy vs. numpy, etc.). Should try to do this without changing anything substantive about the code (i.e. don't change from quantile to kmeans) and then see how fast it runs.</span>
2. <span style="text-decoration: line-through;">Create jupyter notebook page that has the above summary by chunk and the actual code below the summary 
3. <span style="text-decoration: line-through;">Write up verbal explanation of what the general point of the code is. Explain key ideas: a) need to get coordinates for anchor countries, which requires that we have estimated coordinates for Russia (and why we have to estimate a point, as opposed to the whole distribution, which is what we do for the non-anchor countries) b) why we want to use the wasserstein distance between countries (instead of just estimating optimal single points), how wasserstein distance is computationally expensive so we have to discretize the distribution (and how this is arguably the key step in the whole process), etc. c)   
5. <span style="text-decoration: line-through;">Test various discretization/marginal definition methods. Specifically, try: quantile vs. k-means, uniform vs. non-uniform (TBD) marginal definition. Do this by trying each discretized method and then compare to the sample gotten via rejection sampling (which is too big, which is why we can't use it ourselves)</span>
6. Walk through step-by-step example of finding the empirical distribution of a single country, using figures to show how the quantization process works, etc. 
8. discuss the optimal sigma stuff 
9. Discuss how we fill in the missing US data for 2008 and 2012 
