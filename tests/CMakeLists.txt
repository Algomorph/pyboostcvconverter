
add_library(test_project SHARED test_project_source.cpp)

target_link_libraries(test_project static_pbcvt)
target_include_directories(test_project PRIVATE ${PROJECT_SOURCE_DIR}/include)
get_target_property(TEST_OUT_PATH test_project LIBRARY_OUTPUT_NAME)

set_target_properties(test_project PROPERTIES
                      ARCHIVE_OUTPUT_NAME test_project_${ARCHIVE_OUTPUT_SUFFIX}  # prevent name conflict for python2/3 outputs
                      PREFIX ""
                      OUTPUT_NAME test_project
                      SUFFIX ${${PROJECT_NAME}_PY_SUFFIX})

add_custom_command(TARGET test_project POST_BUILD
                  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:test_project> "${CMAKE_CURRENT_SOURCE_DIR}/../tests/$<TARGET_FILE_NAME:test_project>"
)