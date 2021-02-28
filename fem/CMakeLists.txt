cmake_minimum_required(VERSION 2.8.8)

find_package(deal.II 9.2.0 REQUIRED HINTS /usr/local)
DEAL_II_INITIALIZE_CACHED_VARIABLES()

project(fem)

add_executable(
    fem 
    main.cpp 
    material_model.h
    material_model.cpp 
    model.h 
    model.cpp 
    point_history.h 
    point_history.cpp 
)
DEAL_II_SETUP_TARGET(fem)