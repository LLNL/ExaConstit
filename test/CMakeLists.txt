#------------------------------------------------------------------------------
# Dependencies
#------------------------------------------------------------------------------
set(EXACONSTIT_TEST_DEPENDS)

exaconstit_fill_depends_list(LIST_NAME  EXACONSTIT_TEST_DEPENDS
                             DEPENDS_ON  mfem ecmech raja mpi snls)

if(ENABLE_OPENMP)
    list(APPEND EXACONSTIT_TEST_DEPENDS openmp)
endif()

if(ENABLE_CUDA)
    list(APPEND EXACONSTIT_TEST_DEPENDS cuda)
endif()

list(APPEND EXACONSTIT_TEST_DEPENDS exaconstit_static)

blt_add_executable(NAME       test_pa
                   SOURCES    mechanics_test.cpp
                   OUTPUT_DIR ${TEST_OUTPUT_DIR}
                   DEPENDS_ON ${EXACONSTIT_TEST_DEPENDS} gtest)

blt_add_test(NAME    test_partial_assembly
             COMMAND test_pa)