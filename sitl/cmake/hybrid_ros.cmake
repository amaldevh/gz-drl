# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

# ============================================================================
# ROS Detection and Setup
# ============================================================================
if(ROS_VER EQUAL 1)
    message(STATUS "Configuring for ROS 1 (using pkg-config, no catkin build system)")

    # Use ONLY pkg-config for ROS 1 packages to avoid catkin build infrastructure
    # The ROS CMake configs (roscppConfig.cmake etc) load catkin which generates
    # install rules we don't want. pkg-config gives us just the compile/link flags.
    
    find_package(PkgConfig REQUIRED)
    
    # Store ROS-specific variables
    set(ROS_INCLUDE_DIRS "")
    set(ROS_LIBRARIES "")
    set(ROS_LIBRARY_DIRS "")
    set(ROS_DEFINITIONS "")
    
    # Find each ROS package using pkg-config only
    foreach(pkg ${ROS_PKGS})
        pkg_check_modules(${pkg} REQUIRED ${pkg})
        if(${pkg}_INCLUDE_DIRS)
            list(APPEND ROS_INCLUDE_DIRS ${${pkg}_INCLUDE_DIRS})
        endif()
        if(${pkg}_LIBRARIES)
            list(APPEND ROS_LIBRARIES ${${pkg}_LIBRARIES})
        endif()
        if(${pkg}_LIBRARY_DIRS)
            list(APPEND ROS_LIBRARY_DIRS ${${pkg}_LIBRARY_DIRS})
        endif()
        if(${pkg}_CFLAGS_OTHER)
            list(APPEND ROS_DEFINITIONS ${${pkg}_CFLAGS_OTHER})
        endif()
    endforeach()
    
    # Remove duplicates
    if(ROS_INCLUDE_DIRS)
        list(REMOVE_DUPLICATES ROS_INCLUDE_DIRS)
    endif()
    if(ROS_LIBRARIES)
        list(REMOVE_DUPLICATES ROS_LIBRARIES)
    endif()
    if(ROS_LIBRARY_DIRS)
        list(REMOVE_DUPLICATES ROS_LIBRARY_DIRS)
    endif()
    
    # Add ROS includes to the project
    include_directories(
        include
        ${ROS_INCLUDE_DIRS}
    )
    
elseif(ROS_VER EQUAL 2)
    message(STATUS "Configuring for ROS 2")

    # Find ament_cmake
    find_package(ament_cmake REQUIRED)
    
    # Find all requested ROS 2 packages
    set(ROS_DEPENDENCIES "")
    if(ROS_PKGS)
        foreach(pkg ${ROS_PKGS})
            find_package(${pkg} REQUIRED)
            list(APPEND ROS_DEPENDENCIES ${pkg})
        endforeach()
    endif()
    
    # Store ROS-specific variables (aggregate from all packages)
    set(ROS_INCLUDE_DIRS "")
    set(ROS_LIBRARIES "")
    set(ROS_LIBRARY_DIRS "")
    set(ROS_DEFINITIONS "")
    
    # Collect include directories and libraries from all found packages
    foreach(pkg ${ROS_DEPENDENCIES})
        if(${pkg}_INCLUDE_DIRS)
            list(APPEND ROS_INCLUDE_DIRS ${${pkg}_INCLUDE_DIRS})
        endif()
        if(${pkg}_LIBRARIES)
            list(APPEND ROS_LIBRARIES ${${pkg}_LIBRARIES})
        endif()
        if(${pkg}_LIBRARY_DIRS)
            list(APPEND ROS_LIBRARY_DIRS ${${pkg}_LIBRARY_DIRS})
        endif()
        if(${pkg}_DEFINITIONS)
            list(APPEND ROS_DEFINITIONS ${${pkg}_DEFINITIONS})
        endif()
    endforeach()
    
    # Remove duplicates
    if(ROS_INCLUDE_DIRS)
        list(REMOVE_DUPLICATES ROS_INCLUDE_DIRS)
    endif()
    if(ROS_LIBRARIES)
        list(REMOVE_DUPLICATES ROS_LIBRARIES)
    endif()
    if(ROS_LIBRARY_DIRS)
        list(REMOVE_DUPLICATES ROS_LIBRARY_DIRS)
    endif()
    
    # Add project include directory
    include_directories(include)
    
else()
    message(FATAL_ERROR "Invalid ROS_VER: ${ROS_VER}. Must be 1 or 2.")
endif()

# ============================================================================
# Macro: link_target_with_ros
# ============================================================================
# Links a target with all necessary ROS dependencies
# Usage: link_target_with_ros(my_target)
macro(link_target_with_ros TARGET)
    if(NOT TARGET ${TARGET})
        message(FATAL_ERROR "Target '${TARGET}' does not exist. Create it before calling link_target_with_ros().")
    endif()
    
    message(STATUS "Linking target '${TARGET}' with ROS ${ROS_VER}")
    
    if(ROS_VER EQUAL 1)
        # ROS 1 specific linking
        target_include_directories(${TARGET} PUBLIC
            ${ROS_INCLUDE_DIRS}
        )
        
        if(ROS_LIBRARY_DIRS)
            target_link_directories(${TARGET} PUBLIC
                ${ROS_LIBRARY_DIRS}
            )
        endif()
        
        target_link_libraries(${TARGET} PUBLIC
            ${ROS_LIBRARIES}
        )
        
        if(ROS_DEFINITIONS)
            target_compile_definitions(${TARGET} PUBLIC
                ${ROS_DEFINITIONS}
            )
        endif()
        
        # Add dependency on catkin exported targets
        if(catkin_EXPORTED_TARGETS)
            add_dependencies(${TARGET} ${catkin_EXPORTED_TARGETS})
        endif()
        
    elseif(ROS_VER EQUAL 2)
        # ROS 2 specific linking
        target_include_directories(${TARGET} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
            ${ROS_INCLUDE_DIRS}
        )
        
        if(ROS_LIBRARY_DIRS)
            target_link_directories(${TARGET} PUBLIC
                ${ROS_LIBRARY_DIRS}
            )
        endif()
        
        # Link against ROS 2 libraries
        if(ROS_LIBRARIES)
            target_link_libraries(${TARGET} PUBLIC
                ${ROS_LIBRARIES}
            )
        endif()
        
        # Use ament_target_dependencies for proper dependency handling
        # if(ROS_DEPENDENCIES)
        #     target_link_libraries(${TARGET} PRIVATE ${ROS_DEPENDENCIES})
        #     message(FATAL_ERROR ${ROS_DEPENDENCIES})
        # endif()
        
        if(ROS_DEFINITIONS)
            target_compile_definitions(${TARGET} PUBLIC
                ${ROS_DEFINITIONS}
            )
        endif()
    endif()
    
    # Add ROS version definition
    target_compile_definitions(${TARGET} PUBLIC
        ROS_VER=${ROS_VER}
    )
endmacro()
