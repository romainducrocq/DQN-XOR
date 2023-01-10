# [C++] Template / Example for the GTest framework

- Gtest documenation : http://google.github.io/googletest/reference/testing.html
- Gtest source : https://github.com/google/googletest

****

## How to _

`cd bin`

### Install
`./make.sh` or  
```
$ sudo apt-get update
$ sudo apt-get install g++ cmake valgrind libgtest-dev

$ cd /usr/src/gtest
$ sudo cmake CMakeLists.txt
$ sudo make
$ sudo cp -v lib/*.a /usr/lib
```

### Run
`./app.sh` or  
<span id="how_to_run">
```
$ ./build.sh -R
$ ./compile.sh
$ ./run.sh -m test 
$ ./run.sh -m main 
```
</span>

### Main
`./main.sh [-D -R] <args>` or  
<span id="how_to_main">
```
* Debug   : $ ./main.sh -D 
* Release : $ ./main.sh -R 
```
</span>

### Test
`./test.sh [-D -R] <args>` or  
<span id="how_to_test">
```
* Debug   : $ ./test.sh -D 
* Release : $ ./test.sh -R 
```
</span>

### Memcheck
`./memcheck.sh <cmd>`  
```
* Example : $ ./memcheck.sh ./main.sh 
* Example : $ ./memcheck.sh ./test.sh
```

### Log
`./log.sh [-V] <cmd>`  
```
* Example : $ ./log.sh ./main.sh
* Example : $ ./log.sh -V ./test.sh
```

### Readme
`./readme.sh`  
```
* Update : $ ./readme.sh
```

### Help
`./help.sh`  
```
usage: apps/exec [-h] [-m MOD]

Template GTest

optional args:
  -h      Print help and exit
  -m MOD  Set mode < main | test >
```

****

## Demo

`cd bin && ./app.sh`

```
[==========] Running 11 tests from 5 test suites.
[----------] Global test environment set-up.
[----------] 1 test from AccountTest
[ RUN      ] AccountTest.BankAccountStartsEmpty
[       OK ] AccountTest.BankAccountStartsEmpty (0 ms)
[----------] 1 test from AccountTest (0 ms total)

[----------] 2 tests from BankAccountTest
[ RUN      ] BankAccountTest.BankAccountStartsEmpty_F
[       OK ] BankAccountTest.BankAccountStartsEmpty_F (0 ms)
[ RUN      ] BankAccountTest.CanDepositMoney_F
[       OK ] BankAccountTest.CanDepositMoney_F (0 ms)
[----------] 2 tests from BankAccountTest (0 ms total)

[----------] 2 tests from MatrixInstanciation/MatrixTest
[ RUN      ] MatrixInstanciation/MatrixTest.Matrix_dot_P/0
[       OK ] MatrixInstanciation/MatrixTest.Matrix_dot_P/0 (0 ms)
[ RUN      ] MatrixInstanciation/MatrixTest.Matrix_dot_P/1
[       OK ] MatrixInstanciation/MatrixTest.Matrix_dot_P/1 (0 ms)
[----------] 2 tests from MatrixInstanciation/MatrixTest (0 ms total)

[----------] 4 tests from ActivationInstanciation/ActivationTest
[ RUN      ] ActivationInstanciation/ActivationTest.Activation_func_P/0
[       OK ] ActivationInstanciation/ActivationTest.Activation_func_P/0 (0 ms)
[ RUN      ] ActivationInstanciation/ActivationTest.Activation_func_P/1
[       OK ] ActivationInstanciation/ActivationTest.Activation_func_P/1 (0 ms)
[ RUN      ] ActivationInstanciation/ActivationTest.Activation_func_P/2
[       OK ] ActivationInstanciation/ActivationTest.Activation_func_P/2 (0 ms)
[ RUN      ] ActivationInstanciation/ActivationTest.Activation_func_P/3
[       OK ] ActivationInstanciation/ActivationTest.Activation_func_P/3 (0 ms)
[----------] 4 tests from ActivationInstanciation/ActivationTest (0 ms total)

[----------] 2 tests from WithdrawAccountInstanciation/WithdrawAccountTest
[ RUN      ] WithdrawAccountInstanciation/WithdrawAccountTest.FinalBalance_P/0
[       OK ] WithdrawAccountInstanciation/WithdrawAccountTest.FinalBalance_P/0 (0 ms)
[ RUN      ] WithdrawAccountInstanciation/WithdrawAccountTest.FinalBalance_P/1
[       OK ] WithdrawAccountInstanciation/WithdrawAccountTest.FinalBalance_P/1 (0 ms)
[----------] 2 tests from WithdrawAccountInstanciation/WithdrawAccountTest (0 ms total)

[----------] Global test environment tear-down
[==========] 11 tests from 5 test suites ran. (0 ms total)
[  PASSED  ] 11 tests.

-------------------------------MAIN--------------------------------

[INFO]  Hello world!
Time: 0.012403ms
```

****

@romainducrocq
