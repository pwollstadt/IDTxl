# IDTxl

The **I**nformation **D**ynamics **T**oolkit **xl** (IDTxl) in Python. IDTxl is a comprehensive software package for

1. Efficient inference of networks from multivariate time series data using information theory. We utilises a greedy or iterative approach with multivariate transfer entropy for building sets of parent sources for each target node in the network. This iterative conditioning is designed to both remove redundancies and capture synergistic interactions in building each parent set. Rigorous statistical controls (based on comparison to null distributions from time-series surrogates) are used to gate parent selection and to provide automatic stopping conditions for the inference.

2. Analysis of information dynamics
	- multivariate transfer entropy (mTE)
	- mutual information (MI)
	- bivariate transfer entropy (bTE)
	- active information storage (AIS)
	- partial information decomposition (PID)

To **get started** have a look at the [wiki](https://github.com/pwollstadt/IDTxl/wiki) and the [documentation](http://pwollstadt.github.io/IDTxl/).

## How to cite
Wollstadt, Lizier, Vicente, Finn, Martinez Zarzeula, Lindner, Martinez Mediano, Novelli, Wibral, 2017. "IDTxl - The Information Dynamics Toolkit xl: a Python package for the efficient analysis of multivariate information dynamics in networks", GitHub Repository: https://github.com/pwollstadt/IDTxl.

## Contributors

- [Patricia Wollstadt](http://patriciawollstadt.de/), Brain Imaging Center, MEG Unit, Goethe-University, Frankfurt, Germany
- [Michael Wibral](http://www.michael-wibral.de/), Brain Imaging Center, MEG Unit, Goethe-University, Frankfurt, Germany
- [Joseph T. Lizier](http://lizier.me/joseph/), Centre for Complex Systems, The University of Sydney, Sydney, Australia
- [Raul Vicente](http://neuro.cs.ut.ee/people/), Computational Neuroscience Lab, Institute of Computer Science, University of Tartu, Tartu, Estonia
- Conor Finn, Centre for Complex Systems, The University of Sydney, Sydney, Australia
- Mario Martínez Zarzuela, Department of Signal Theory and Communications and Telematics Engineering, University of Valladolid, Valladolid, Spain
- [Michael Lindner](https://www.reading.ac.uk/Psychology/About/staff/m-lindner.aspx), Center for Integrative Neuroscience and Neurodynamics, University of Reading, Reading, UK
- Leonardo Novelli, Centre for Complex Systems, The University of Sydney, Sydney, Australia
- [Pedro Martinez Mediano](https://www.doc.ic.ac.uk/~pam213/), Computational Neurodynamics Group, Imperial College London, London, United Kingdom

## Acknowledgements

This project has been supported by funding through:

- Universities Australia - Deutscher Akademischer Austauschdienst (German Academic Exchange Service) UA-DAAD Australia-Germany Joint Research Co-operation grant "Measuring neural information synthesis and its impairment", Wibral, Lizier, Priesemann, Wollstadt, Finn, 2016-17
- Australian Research Council Discovery Early Career Researcher Award (DECRA) "Relating function of complex networks to structure using information theory", Lizier, 2016-19

## References
+ Multivariate transfer entropy: *Lizier & Rubinov, 2012, Preprint, Technical Report 25/2012,
Max Planck Institute for Mathematics in the Sciences. Available from:
http://www.mis.mpg.de/preprints/2012/preprint2012_25.pdf*
+ Kraskov estimator: *Kraskov et al., 2004, Phys Rev E 69, 066138*
+ Nonuniform embedding: *Faes et al., 2011, Phys Rev E 83, 051112*
+ Faes' compensated transfer entropy: *Faes et al., 2013, Entropy 15, 198-219*
