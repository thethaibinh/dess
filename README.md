# Depth-based Sampling and Steering Constraints for Memoryless Local Planners

This repository contains source code implementing the depth-based algorithm described in the paper "DEpth-based Sampling and Steering Constraints for Memoryless Local Planners", which has been submitted to a journal and is under review. Please don't hesitate to contact the corresponding author [Thai Binh Nguyen](mailto:thethaibinh@gmail.com) if you have any requests.

## Acknowledgements
This project is inspired by [RAPPIDS](https://github.com/nlbucki/RAPPIDS) from [Nathan Bucki](mailto:nathan_bucki@berkeley.edu) et al. We use some components from RAPPIDS.
## Getting Started

First clone the repository and enter the created folder:
```
git clone https://github.com/thethaibinh/DESS
cd DESS
```

Create a build folder and compile:
```
mkdir build
cd build
cmake ..
make
```
A program is provided that demonstrates the performance of the algorithm and gives an example of how the algorithm can be used to generate best-cost collision free motion primitives. The program `Benchmarker` performs the Monte Carlo simulations described in Seciton IV-B of the associated paper. The test performed in the paper can be ran from the dess folder with the following commands:
```
./build/test/Benchmarker -n 10000 --w 640 --h 480 --f 386 --cx 320 --cy 240 --numCompTimesForTCTest 15 --maxCompTimeForTCTest 0.02
```
Note the `-n` option can be changed to a smaller number to perform less Monte Carlo trials, and thus run the tests faster (but with less accuracy). The above settings reflect those used to generate the results reported in the paper.

The test generates a `.json` file in the data folder containing the test results. We provide three python scripts to visualize the results of the overall planner performance test. They can be ran with the following commands:
```
cd scripts
python plotAvgTrajGenNum.py
python plotNumCollisionFree.py
python plotBestCost.py
```

## Documentation
An HTML file generated with [Doxygen](http://www.doxygen.nl/) can be accessed after cloning the repository by opening `Documentation.html` in the `doc/` folder.

## Licensing

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.
