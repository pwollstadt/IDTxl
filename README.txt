# IDTxl

The **I**nformation **D**ynamics **T**oolkit **xl** (IDTxl) is a comprehensive software
package for efficient inference of networks and their node dynamics from
multivariate time series data using information theory. IDTxl provides
functionality to estimate the following measures:

1) For network inference:
    - multivariate transfer entropy (TE)/Granger causality (GC)
    - multivariate mutual information (MI)
    - bivariate TE/GC
    - bivariate MI
2) For analysis of node dynamics:
    - active information storage (AIS)
    - partial information decomposition (PID)

IDTxl implements estimators for discrete and continuous data with parallel
computing engines for both GPU and CPU platforms. Written for Python3.4.3+.

To **get started** have a look at the [wiki](https://github.com/pwollstadt/IDTxl/wiki) and the [documentation](http://pwollstadt.github.io/IDTxl/). For further discussions, join [IDTxl's google group](https://groups.google.com/forum/#!forum/idtxl).

## How to cite
WP. Wollstadt, J. T. Lizier, R. Vicente, C. Finn, M. Martinez-Zarzuela, P. Mediano, L. Novelli, M. Wibral (2018). _IDTxl: The Information Dynamics Toolkit xl: a Python package for the efficient analysis of multivariate information dynamics in networks._ ArXiv preprint: https://arxiv.org/abs/1807.10459.

## Contributors

- [Patricia Wollstadt](http://patriciawollstadt.de/), Brain Imaging Center, MEG Unit, Goethe-University, Frankfurt, Germany
- [Michael Wibral](http://www.michael-wibral.de/), Brain Imaging Center, MEG Unit, Goethe-University, Frankfurt, Germany
- [Joseph T. Lizier](http://lizier.me/joseph/), Centre for Complex Systems, The University of Sydney, Sydney, Australia
- [Raul Vicente](http://neuro.cs.ut.ee/people/), Computational Neuroscience Lab, Institute of Computer Science, University of Tartu, Tartu, Estonia
- Conor Finn, Centre for Complex Systems, The University of Sydney, Sydney, Australia
- Mario Martinez-Zarzuela, Department of Signal Theory and Communications and Telematics Engineering, University of Valladolid, Valladolid, Spain
- Leonardo Novelli, Centre for Complex Systems, The University of Sydney, Sydney, Australia
- [Pedro Mediano](https://www.doc.ic.ac.uk/~pam213/), Computational Neurodynamics Group, Imperial College London, London, United Kingdom

**How to contribute?** We are happy about any feedback on IDTxl. If you would like to contribute, please open an issue or send a pull request with your feature or improvement. Also have a look at the [developer's section](https://github.com/pwollstadt/IDTxl/wiki#developers-section) in the Wiki for details.


## Acknowledgements

This project has been supported by funding through:

- Universities Australia - Deutscher Akademischer Austauschdienst (German Academic Exchange Service) UA-DAAD Australia-Germany Joint Research Co-operation grant "Measuring neural information synthesis and its impairment", Wibral, Lizier, Priesemann, Wollstadt, Finn, 2016-17
- Australian Research Council Discovery Early Career Researcher Award (DECRA) "Relating function of complex networks to structure using information theory", Lizier, 2016-19

## Key References
+ Multivariate transfer entropy: *Lizier & Rubinov, 2012, Preprint, Technical Report 25/2012,
Max Planck Institute for Mathematics in the Sciences. Available from:
http://www.mis.mpg.de/preprints/2012/preprint2012_25.pdf*
+ Kraskov estimator: *Kraskov et al., 2004, Phys Rev E 69, 066138*
+ Nonuniform embedding: *Faes et al., 2011, Phys Rev E 83, 051112*
+ Faes' compensated transfer entropy: *Faes et al., 2013, Entropy 15, 198-219*
+ PID: *Williams & Beer, 2010, arXiv preprint: http://arxiv.org/abs/1004.2515*
+ PID estimators: *Bertschinger et al., 2014, Entropy, 16(4); Makkeh et al., 2017, Entropy, 19(10),
  Makkeh et al., 2018, Entropy, 20(271)*
