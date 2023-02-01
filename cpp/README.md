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
- [Ensure Conan is installed on your system](https://tek5030.github.io/tutorial/conan.html), unless you are not on a lab computer.
- Install project dependencies using conan:

   ```bash
   # git clone https://github.com/tek5030/lab-transformations.git
   # cd lab-transformations/cpp

   conan install . --install-folder=build --build=missing
   ```
- When you configure the project in CLion, remember to set `build` as the _Build directory_, as described in [lab 0](https://github.com/tek5030/lab_00/blob/master/cpp/lab-guide/1-open-project-in-clion.md#6-configure-project).