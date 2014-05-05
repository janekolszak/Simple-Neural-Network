**Discontinued!** Check https://github.com/janekolszak/graph-tool-nn for a python3.x library.

# Simple Neural Network

Simple Neural Network (SNN) is a <b>C++</b> library (<b>Python bindings planned</b>) with an implementation of artificial neural networks. The main objective is to deliver an understandable but powerful library, which could be easily modified by <b>anyone</b> .

<b><i>If <b>you</b> want to implement a well known algorithm or develop and test your own this is a perfect place for you!</i></b>

SNN depends on some <i>boost</i> libs and is shipped with GPLv2 license.

# Building
This project requires:
* boost::log
* boost::python
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

# Implementation guidelines

OK, so you want to expand SNN' functionality! Nice! It can be for your purpose only or you gonna make a pull request later on, it's up to you!

Some things you should keep in mind:
* We intentionally don't declare private class members to make writing code easier, but you should still avoid spaghetti code!
* If you implement another learning algorithm, keep it in an Neuron. Just declare another kind of Neuron and declare a <i>learn(...)</i> method. It's all in implementation folder.
