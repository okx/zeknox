# FAQ

This is a non-exhaustive list of errors that user can encounter and how to solve them.

Note: all these issues and solutions apply to Ubuntu 22.04 or 24.04.

1. ``unsupported display driver / cuda driver combination``

Solution: install the latest Nvidia driver and utilities, then reboot your system. For example:
```
sudo apt install -y nvidia-driver-565 nvidia-utils-565
sudo apt reboot
```

2. When running ``sudo make install`` in ``native/build``, I get this error:
```
CMake Error at cmake_install.cmake:70 (file):
  file INSTALL cannot find
  ".../native/depends/blst/libblst.a": No such file
```

Solution:
```
cd native/depends/blst
./build.sh
```

3. When running Rust tests, I get:
```
NUM_OF_GPUS should be set: NotPresent
```

Solution: ``export NUM_OF_GPUS=1`` (or the number of GPUs in your system).