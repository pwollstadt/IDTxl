# Dev folder

Folder contains WIP and deliverables, as well as UML diagrams of the current release. 

# Creating UML diagrams from Python code using pylint

[pyreverse](https://www.logilab.org/blogentry/6883) is part of pylint package and can be used to automatically generate UML diagrams from Python code (requires a Graphviz installation). To use pyreverse, simply install pylint. 

By default, pyreverse creates two diagrams showing the packages and classes belonging to a project. Call:

```
$ pyreverse -o svg -p idtxl PATH_TO_IDTXL/IDTxl/idtxl
parsing ...
```

to create `classes_idtxl.svg` and `packages_idtxl.svg` in the current folder, where `-o`sets the output format and `-p` sets the project name. Add the `-f ALL` to also include private methods and attributes (default=`PUB_ONLY: filter all non public attributes`). See also the [pyreverse documentation](https://docs.oracle.com/cd/E36784_01/html/E36870/pyreverse-1.html).
