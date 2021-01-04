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
## Borrowed from Conduit https://github.com/LLNL/conduit
## The license file can be found under 
##------------------------------------------------------------------------------
## - Adds a python based unit test
##
## add_python_test( TEST test)
##------------------------------------------------------------------------------
function(add_python_test )
    set(options)
    set(singleValueArgs TEST )
    set(multiValueArgs )

    # parse our arguments
    cmake_parse_arguments(args
                         "${options}"
                         "${singleValueArgs}"
                         "${multiValueArgs}" ${ARGN} )

    message(STATUS " [*] Adding Python-based Unit Test: ${args_TEST}")
    add_test(NAME ${args_TEST} COMMAND
             ${PYTHON_EXECUTABLE} ${args_TEST})

    # use proper env var path sep for current platform
    if(WIN32)
        set(ENV_PATH_SEP "\\;")
    else()
        set(ENV_PATH_SEP ":")
    endif()
    # make sure python can pick up the modules we built
    # if python path is already set -- we need to append to it
    # this is important for running in spack's build-env
    set(py_path "")
    if(DEFINED ENV{PYTHONPATH})
        set(py_path "$ENV{PYTHONPATH}${ENV_PATH_SEP}")
    endif()
    set_property(TEST ${args_TEST}
                 PROPERTY
                 ENVIRONMENT "PYTHONPATH=${py_path}${CMAKE_BINARY_DIR}/python-modules/${ENV_PATH_SEP}${CMAKE_CURRENT_SOURCE_DIR}")
    if(WIN32)
        # proper path to dlls for vstudio and proper path for non config based gen (nmake, etc)
        set_property(TEST ${args_TEST}
                     APPEND
                     PROPERTY
                     ENVIRONMENT "PATH=${CMAKE_BINARY_DIR}/bin/${ENV_PATH_SEP}${CMAKE_BINARY_DIR}/bin/$<CONFIG>/${ENV_PATH_SEP}$ENV{PATH}")
    endif()

    # set folder if passed
    if( DEFINED args_FOLDER )
        blt_set_target_folder(TARGET ${args_TEST} FOLDER ${args_FOLDER})
    endif()

endfunction(add_python_test)

####################################
# Add Python Module Tests
####################################
set(PYTHON_MODULE_TESTS test_mechanics.py)


foreach(TEST ${PYTHON_MODULE_TESTS})
    add_python_test(TEST ${TEST}
                    FOLDER ${TEST_OUTPUT_DIR})
endforeach()

# A hack more or less to get the test config files into the problem
add_custom_command(TARGET test_pa PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory
                       ${CMAKE_SOURCE_DIR}/test/data/ $<TARGET_FILE_DIR:test_pa>/../test/)

add_custom_command(TARGET test_pa POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${CMAKE_SOURCE_DIR}/test/test_mechanics.py $<TARGET_FILE_DIR:test_pa>/../test/test_mechanics.py
)

#add_test(NAME    test_python
#         COMMAND ${PYTHON_EXECUTABLE} test_mechanics.py)
