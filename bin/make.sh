#!/bin/bash

sudo apt-get update
sudo apt-get install g++ cmake valgrind libgtest-dev

wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
rm -v libtorch-shared-with-deps-latest.zip
sudo rm -rv /usr/lib/libtorch
sudo mv -v libtorch /usr/lib/

cd /usr/src/gtest
sudo cmake CMakeLists.txt
sudo make
sudo cp -v lib/*.a /usr/lib
