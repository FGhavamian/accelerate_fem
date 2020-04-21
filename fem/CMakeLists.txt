CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.5.0 REQUIRED
        HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR} /usr/local/bin/deal.ii
        )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(accelerate_simulations)

ADD_EXECUTABLE(accelerate_simulations main.cpp)
DEAL_II_SETUP_TARGET(accelerate_simulations)