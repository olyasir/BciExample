cmake_minimum_required(VERSION 3.25)
project(bci_example LANGUAGES CXX C)


set(CMAKE_BUILD_TYPE Debug)

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 ${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS "-fPIC -std=c++20 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_C_FLAGS_DEBUG "-g -O0 ${CMAKE_C_FLAGS_DEBUG}")
  set(CMAKE_C_FLAGS "-fPIC ${CMAKE_C_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od /DEBUG ${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS "/std:c++20 ${CMAKE_CXX_FLAGS}")
endif()


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER ON CACHE BOOL "")
if(APPLE)
  set(USE_METAL ON CACHE BOOL "Use the Metal API for TVM runtime")
else()
  set(USE_VULKAN ON CACHE BOOL "Use the Vulkan API for TVM runtime")
endif()


# Add mlc-llm subdirectory
add_subdirectory(mlc-llm)


message(PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR})

set(MLC_LLM_HOME ${PROJECT_SOURCE_DIR}/mlc-llm)


set(TVM_HOME ${MLC_LLM_HOME}/3rdparty/tvm)
message(TVM_HOME=${TVM_HOME})

set(TVM_INCLUDES ${TVM_HOME}/include
                 ${TVM_HOME}/3rdparty/dlpack/include
                 ${TVM_HOME}/3rdparty/dmlc-core/include
                 ${TVM_HOME}/3rdparty/picojson)
set(TOKENIZERS_INCLUDES ${MLC_LLM_HOME}/3rdparty/tokenizers-cpp/include)


set(LIB_NAME bci_model)
set(LIB_SOURCES include/bci_model.h
                src/bci_model.cc
                )
add_library(${LIB_NAME} STATIC ${LIB_SOURCES})
add_executable(cli example.cc)

include_directories(${PROJECT_SOURCE_DIR}/include)

target_include_directories(${LIB_NAME} PUBLIC ${MLC_LLM_HOME}
                                                 ${TVM_INCLUDES}
                                                 ${TOKENIZERS_INCLUDES})

target_include_directories(cli PRIVATE ${PROJECT_SOURCE_DIR} 
                                        ${PROJECT_SOURCE_DIR}/include
                                        ${MLC_LLM_HOME}
                                        ${TVM_INCLUDES}
                                        ${TOKENIZERS_INCLUDES})


# Include directories
target_link_libraries(${LIB_NAME} PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,mlc_llm_static,tvm_runtime,tokenizers_cpp>")
target_link_libraries(cli PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LIB_NAME}>")
