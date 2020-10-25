# geoemtry

## make mesh with GMSH

* Install pygmsh: <https://github.com/nschloe/pygmsh/blob/master/README.md>
  * copy `gmsh.py` from `/usr/lib/python3/dist-packages/gmsh.py` to `site-packages` direcotry of your virtual environment.

* convert msh file to quads using (git clone <https://github.com/martemyev/tethex.git>)
  * clone the repo
  * `mkdir build`
  * `cd build`
  * `cmake ..`
  * `make`
  *  `PATH_TO_BUILD_DIR/tethex INPUT.msh OUTPUT.msh VEROBOSE_INT`