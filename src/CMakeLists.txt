MESSAGE ("Generating makefile for src:")

FILE (GLOB_RECURSE project_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)
MESSAGE ("Files: " ${project_SRCS})

ADD_LIBRARY(${APPLICATION_CODENAME} SHARED ${project_SRCS})

SET (project_LIBS ${Boost_LIBRARIES})
SET (project_BIN ${PROJECT_NAME})