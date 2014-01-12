MESSAGE ("Generating makefile for tests...")

FILE (GLOB_RECURSE tests_SRCS *.cpp *.cxx *.cc *.C *.c *.h *.hpp)
SET (tests_LIBS  ${PROJECT_NAME} ${PYTHON_LIBRARIES} ${Boost_LIBRARIES} )
SET (tests_BIN "tests")

ADD_DEFINITIONS(-DBOOST_TEST_DYN_LINK)

LINK_DIRECTORIES(${MAINFOLDER}/lib)
ADD_EXECUTABLE(${tests_BIN} ${tests_SRCS})
TARGET_LINK_LIBRARIES(${tests_BIN} ${tests_LIBS})
