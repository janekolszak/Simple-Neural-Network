# Simple Neural Network

Simple Neural Network (SNN) is a <b>C++</b> library (<b>Python bindings planned</b>) with an implementation of artificial neural networks. The main objective is to deliver an understandable but powerfull library, which <b>anyone</b> could easily modify. 

<b><i>If <b>you</b> want to implement a well known algorithm or develop and test your own this is a perfect place for you!</i></b>

SNN depends on some <i>boost</i> libs and is shipped with GPLv2 license. 

# Building
This project requires:
* boost::log 
* boost::python 
* boost::math
* libpython2.7

### Basic

    ./configure
    make

Now you have your libsnn in the SNN's <i>./lib</i> directory.

### Custom boost and python libraries paths

You can easily pass the paths to your <i>boost</i> or <i>python</i> libs to the configure script.
Since the configure script is just calling <i>cmake</i> you can use command-line defines like:
* -DBOOST_ROOT=...
* -DBOOST_LIBRARYDIR=...
* -DBOOST_INCLUDEDIR=...
* -DPYTHONLIBS_LIBRARYDIR=...

For example:

    ./configure -DBOOST_ROOT=path/to/boostroot -DPYTHONLIBS_LIBRARYDIR=path/to/libpython2.7.so
    make