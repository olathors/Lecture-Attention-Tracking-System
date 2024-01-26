# Lab 1: Transformations with Eigen
We will in this lab play around with the [Eigen library](http://eigen.tuxfamily.org/), which is a C++ library for linear algebra that we will use throughout this course.

This lab consists of two different code projects, which are both part of this repository.
Start by cloning this repository on your machine.

Then open the lab project in CLion using the cmake-file in the base directory: ```lab-transformations/cpp/CMakeLists.txt``` .
If you are uncertain about how this is done, please take a look at [last week's lab](https://github.com/tek5030/lab_00/blob/master/lab-guide/1-open-project-in-clion.md).
Remember to use the correct repository address!


The lab is carried out by following these steps:
1. [Get to know Eigen](lab-guide/1-get-to-know-eigen.md)
2. [Image transformations with Eigen and OpenCV](lab-guide/2-image-transformations-with-eigen-and-opencv.md)

When appropriate, take a look at [our proposed solution](https://github.com/tek5030/solution-transformations).

Please start the lab by going to the [first step](lab-guide/1-get-to-know-eigen.md).

## Prerequisites
- Eigen must be installed on your system. If you are on a lab computer, you are all set.

   If your are on Ubuntu 22, but not on a lab computer, the following should be sufficient.

   ```bash
   sudo apt update
   sudo apt install libeigen3-dev
   ```
- We refer to [setup_scripts](https://github.com/tek5030/setup_scripts) and [lab-introduction](https://github.com/tek5030/lab_00/blob/master/cpp/lab-guide/1-open-project-in-clion.md#6-configure-project) as a general getting started-guide for the C++ labs on Ubuntu 22.04.