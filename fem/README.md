# Finite element simulation using deall.II

Install cmake: <https://vitux.com/how-to-install-cmake-on-ubuntu-18-04/>

Download deall.II: <https://www.dealii.org/download.html>
Install deal.II: <https://www.dealii.org/current/readme.html#installation>

compile:
<code>cd build</code>
<code>cmake ..</code>
<code>make</code>

run:
<code>build/fem --PATH_MESH --PATH_SOLUTION --PATH_PLASTIC_STRAIN --MAT1_b --MAT1_y --MAT2_b -- MAT2_y</code>