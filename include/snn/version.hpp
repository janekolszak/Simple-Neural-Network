#ifndef SNN_VERSION_HPP
#define SNN_VERSION_HPP


/**
 * SNN_VERSION % 100 is the patch level
 * SNN_VERSION / 100 % 1000 is the minor version
 * SNN_VERSION / 100000 is the major version
 */
#define SNN_VERSION 000100

/**
 * SNN_LIB_VERSION must be defined to be the same as SNN_VERSION
 * but as a *string* in the form "x_y[_z]" where:
 *     x is the major version number, 
 *     y is the minor version number, 
 *     z is the patch level if not 0.
 */
#define SNN_LIB_VERSION "0.1.0"


#endif 
