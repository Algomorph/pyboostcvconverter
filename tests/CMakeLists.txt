
add_library(test_project SHARED test_project_source.cpp)

target_link_libraries(test_project static_pbcvt)
target_include_directories(test_project PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(test_project PROPERTIES
                      LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/tests
                      LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/tests
)
