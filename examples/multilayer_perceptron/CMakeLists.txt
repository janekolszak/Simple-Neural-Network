FILE (GLOB_RECURSE multilayer_perceptron_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)
SET (multilayer_perceptron_LIBS ${Boost_LIBRARIES} ${PROJECT_NAME})
SET (multilayer_perceptron_BIN "multilayer_perceptron_example")

IF (NOT CMAKE_CROSSCOMPILING)

    LINK_DIRECTORIES(${MAINFOLDER}/lib)
    ADD_EXECUTABLE(${multilayer_perceptron_BIN} ${multilayer_perceptron_SRCS})
    TARGET_LINK_LIBRARIES(${multilayer_perceptron_BIN} ${multilayer_perceptron_LIBS})

ENDIF (NOT CMAKE_CROSSCOMPILING)
