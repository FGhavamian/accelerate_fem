# geoemtry

## make mesh with GMSH

In case of this error: `OSError: libGLU.so.1: cannot open shared object file: No such file or directory`

Check this: `https://askubuntu.com/a/1148920/936905`


* convert msh file to quads using (git clone <https://github.com/martemyev/tethex.git>)
  * clone the repo
  * `mkdir build`
  * `cd build`
  * `cmake ..`
  * `make`
  *  `PATH_TO_BUILD_DIR/tethex INPUT.msh OUTPUT.msh VEROBOSE_INT`

