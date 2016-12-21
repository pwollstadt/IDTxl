# IDTxl

The **I**nformation **D**ynamics **T**oolkit **xl** (IDTxl) in Python. IDTxl is a comprehensive software package for the efficient analysis of information dynamics of large data sets.

IDTxl provides estimators for the following information theoretic measures:

- mutual information (MI)
- bivariate transfer entropy (bTE)
- multivariate transfer entropy (mTE)
- Granger causality (GC)
- active information storage (AIS)
- partial information decomposition (PID)

IDTxl uses GPU-accelerated estimators as well as parallel processing and is designed for the application on high-performance-computing clusters.

To **get started** have a look at the [wiki](https://github.com/pwollstadt/IDTxl/wiki) and the [documentation](http://pwollstadt.github.io/IDTxl/).

## Contributors

- [Patricia Wollstadt](http://patriciawollstadt.de/), Brain Imaging Center, MEG Unit, Goethe-University, Frankfurt, Germany
- [Michael Wibral](http://www.michael-wibral.de/), Brain Imaging Center, MEG Unit, Goethe-University, Frankfurt, Germany
- [Joseph T. Lizier](http://lizier.me/joseph/), Complex Systems Research Group, The University of Sydney, Sydney, Australia
- [Raul Vicente](http://neuro.cs.ut.ee/people/), Computational Neuroscience Lab, Institute of Computer Science, University of Tartu, Tartu, Estonia
- Connor Finn, Complex Systems Research Group, The University of Sydney, Sydney, Australia
- Mario Martínez Zarzuela, Department of Signal Theory and Communications and Telematics Engineering, University of Valladolid, Valladolid, Spain
- Michael Lindner, Center for Integrative Neuroscience and Neurodynamics, University of Reading, Reading, UK


## References
+ Multivariate transfer entropy: *Lizier & Rubinov, 2012, Preprint, Technical Report 25/2012,
Max Planck Institute for Mathematics in the Sciences. Available from:
http://www.mis.mpg.de/preprints/2012/preprint2012_25.pdf*
+ Kraskov estimator: *Kraskov et al., 2004, Phys Rev E 69, 066138*
+ Nonuniform embedding: *Faes et al., 2011, Phys Rev E 83, 051112*
+ Faes' compensated transfer entropy: *Faes et al., 2013, Entropy 15, 198-219*
+ Uniform embedding: *Takens, 1981, Detecting strange attractors in turbulence (pp. 366-381). Springer Berlin Heidelberg*
+ Ragwitz' criterion: *Ragwitz & Kantz, 2002, Phys Rev E 65, 056201*
