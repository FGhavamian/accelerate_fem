# Finite element simulation using deall.II

* Install cmake: <https://vitux.com/how-to-install-cmake-on-ubuntu-18-04/>
* Download deall.II: <https://www.dealii.org/download.html>
* Install deal.II: <https://www.dealii.org/current/readme.html#installation>

* compile:
  * `cd build`
  * `cmake ..`
  * `make`

* run:
  * `build/fem --PATH_MESH --PATH_SOLUTION --PATH_PLASTIC_STRAIN --MAT1_b --MAT1_y --MAT2_b --MAT2_y --N_TIMESTEPS --VERBOSE`

Hint:

* if cmake generated: `cmake: error while loading shared libraries: libssl.so.1.0.0: cannot open shared object file: No such file or directory` error:
  * <https://askubuntu.com/a/1264351/936905>

* in case of error: `*** No rule to make target '/usr/lib/FILE.so'`:
  * `apt-file search '/usr/lib/FILE.so`
  * `sudo apt-get install LIBRARY`

* some missing libraries:
  * `sudo apt-get install libboost-dev`
  * `sudo apt-get install tbb`
  * `sudo apt install zlib1g-dev`
  * `sudo apt install libboost-iostreams1.71-dev`
  * `sudo apt install libboost-serialization1.71-dev`
  * `sudo apt install libscotchmetis-dev`