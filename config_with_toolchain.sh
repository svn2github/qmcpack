#!/bin/env bash
#usage build.sh toolchain
cd build; 
cmake -DCMAKE_TOOLCHAIN_FILE=../config/$1 ..; \
cmake -DCMAKE_TOOLCHAIN_FILE=../config/$1 ..; \
make -j8; cd -
