# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Amal Dev Haridevan

# ============================================================================
# GeneratePackageXML.cmake
# 
# CMake module for automatically generating package.xml files for ROS 1 and ROS 2
# ============================================================================

# ============================================================================
# Macro: generate_ros_package_xml
# ============================================================================
# Automatically generates a package.xml file based on CMake configuration
#
# Arguments (all optional, with defaults):
#   PACKAGE_NAME          - Package name (default: ${PROJECT_NAME})
#   VERSION               - Package version (default: ${PROJECT_VERSION} or "1.0.0")
#   DESCRIPTION           - Package description (default: "Package ${PROJECT_NAME}")
#   MAINTAINER_NAME       - Maintainer name (default: "Developer")
#   MAINTAINER_EMAIL      - Maintainer email (default: "developer@example.com")
#   LICENSE               - License type (default: "MIT")
#   HOMEPAGE_URL          - Project homepage (optional)
#   BUGTRACKER_URL        - Bug tracker URL (optional)
#   REPOSITORY_URL        - Repository URL (optional)
#   AUTHOR_NAME           - Author name (optional, can specify multiple)
#   AUTHOR_EMAIL          - Author email (optional, can specify multiple)
#   BUILD_DEPENDS         - Additional build dependencies (list)
#   EXEC_DEPENDS          - Additional execution dependencies (list)
#   TEST_DEPENDS          - Test dependencies (list)
#   DOC_DEPENDS           - Documentation dependencies (list)
#   BUILDTOOL_DEPENDS     - Additional buildtool dependencies (list)
#   OUTPUT_FILE           - Output file path (default: ${CMAKE_CURRENT_SOURCE_DIR}/package.xml)
#   ROS_VERSION           - ROS version (default: uses global ROS_VERSION variable)
#   ROS_PACKAGES          - ROS packages to depend on (default: uses global ROS_PKGS variable)
#
# Usage:
#   generate_ros_package_xml(
#       PACKAGE_NAME "my_robot_pkg"
#       VERSION "2.1.0"
#       DESCRIPTION "My awesome robot package"
#       MAINTAINER_NAME "John Doe"
#       MAINTAINER_EMAIL "john@example.com"
#       LICENSE "MIT"
#       BUILD_DEPENDS "eigen3" "boost"
#       EXEC_DEPENDS "python3-numpy"
#   )

macro(generate_ros_package_xml)
    # Parse arguments
    set(options "")
    set(oneValueArgs 
        PACKAGE_NAME 
        VERSION 
        DESCRIPTION 
        MAINTAINER_NAME 
        MAINTAINER_EMAIL
        LICENSE
        HOMEPAGE_URL
        BUGTRACKER_URL
        REPOSITORY_URL
        OUTPUT_FILE
        ROS_VERSION
    )
    set(multiValueArgs 
        ROS_PACKAGES
        BUILD_DEPENDS
        EXEC_DEPENDS
        TEST_DEPENDS
        DOC_DEPENDS
        BUILDTOOL_DEPENDS
        AUTHOR_NAME
        AUTHOR_EMAIL
    )
    
    cmake_parse_arguments(PKG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    
    # Set defaults
    if(NOT PKG_PACKAGE_NAME)
        set(PKG_PACKAGE_NAME "${PROJECT_NAME}")
    endif()
    
    if(NOT PKG_VERSION)
        if(PROJECT_VERSION)
            set(PKG_VERSION "${PROJECT_VERSION}")
        else()
            set(PKG_VERSION "1.0.0")
        endif()
    endif()
    
    if(NOT PKG_DESCRIPTION)
        set(PKG_DESCRIPTION "Package ${PKG_PACKAGE_NAME}")
    endif()
    
    if(NOT PKG_MAINTAINER_NAME)
        set(PKG_MAINTAINER_NAME "Developer")
    endif()
    
    if(NOT PKG_MAINTAINER_EMAIL)
        set(PKG_MAINTAINER_EMAIL "developer@example.com")
    endif()
    
    if(NOT PKG_LICENSE)
        set(PKG_LICENSE "MIT")
    endif()
    
    if(NOT PKG_OUTPUT_FILE)
        set(PKG_OUTPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/package.xml")
    endif()
    
    if(NOT PKG_ROS_VERSION)
        set(PKG_ROS_VERSION "${ROS_VERSION}")
    endif()
    
    if(NOT PKG_ROS_PACKAGES)
        set(PKG_ROS_PACKAGES "${ROS_PKGS}")
    endif()
    
    # Validate ROS version
    if(NOT PKG_ROS_VERSION)
        message(FATAL_ERROR "ROS_VERSION must be set (either globally or via ROS_VERSION argument)")
    endif()
    
    if(NOT PKG_ROS_VERSION EQUAL 1 AND NOT PKG_ROS_VERSION EQUAL 2)
        message(FATAL_ERROR "ROS_VERSION must be 1 or 2, got: ${PKG_ROS_VERSION}")
    endif()
    
    # Start building the XML content
    set(XML_CONTENT "<?xml version=\"1.0\"?>\n")
    set(XML_CONTENT "${XML_CONTENT}<?xml-model href=\"http://download.ros.org/schema/package_format")
    
    # Set package format based on ROS version
    if(PKG_ROS_VERSION EQUAL 1)
        set(XML_CONTENT "${XML_CONTENT}2.xsd\" schematypens=\"http://www.w3.org/2001/XMLSchema\"?>\n")
        set(XML_CONTENT "${XML_CONTENT}<package format=\"2\">\n")
    else()
        set(XML_CONTENT "${XML_CONTENT}3.xsd\" schematypens=\"http://www.w3.org/2001/XMLSchema\"?>\n")
        set(XML_CONTENT "${XML_CONTENT}<package format=\"3\">\n")
    endif()
    
    # Basic package information
    set(XML_CONTENT "${XML_CONTENT}  <name>${PKG_PACKAGE_NAME}</name>\n")
    set(XML_CONTENT "${XML_CONTENT}  <version>${PKG_VERSION}</version>\n")
    set(XML_CONTENT "${XML_CONTENT}  <description>${PKG_DESCRIPTION}</description>\n")
    set(XML_CONTENT "${XML_CONTENT}\n")
    
    # Maintainer
    set(XML_CONTENT "${XML_CONTENT}  <maintainer email=\"${PKG_MAINTAINER_EMAIL}\">${PKG_MAINTAINER_NAME}</maintainer>\n")
    
    # Authors (if specified)
    if(PKG_AUTHOR_NAME)
        list(LENGTH PKG_AUTHOR_NAME _author_count)
        list(LENGTH PKG_AUTHOR_EMAIL _email_count)
        
        math(EXPR _max_count "${_author_count} - 1")
        foreach(_idx RANGE ${_max_count})
            list(GET PKG_AUTHOR_NAME ${_idx} _author_name)
            if(_idx LESS ${_email_count})
                list(GET PKG_AUTHOR_EMAIL ${_idx} _author_email)
                set(XML_CONTENT "${XML_CONTENT}  <author email=\"${_author_email}\">${_author_name}</author>\n")
            else()
                set(XML_CONTENT "${XML_CONTENT}  <author>${_author_name}</author>\n")
            endif()
        endforeach()
    endif()
    
    # License
    set(XML_CONTENT "${XML_CONTENT}\n")
    set(XML_CONTENT "${XML_CONTENT}  <license>${PKG_LICENSE}</license>\n")
    set(XML_CONTENT "${XML_CONTENT}\n")
    
    # URLs (if specified)
    if(PKG_HOMEPAGE_URL)
        set(XML_CONTENT "${XML_CONTENT}  <url type=\"website\">${PKG_HOMEPAGE_URL}</url>\n")
    endif()
    if(PKG_BUGTRACKER_URL)
        set(XML_CONTENT "${XML_CONTENT}  <url type=\"bugtracker\">${PKG_BUGTRACKER_URL}</url>\n")
    endif()
    if(PKG_REPOSITORY_URL)
        set(XML_CONTENT "${XML_CONTENT}  <url type=\"repository\">${PKG_REPOSITORY_URL}</url>\n")
    endif()
    if(PKG_HOMEPAGE_URL OR PKG_BUGTRACKER_URL OR PKG_REPOSITORY_URL)
        set(XML_CONTENT "${XML_CONTENT}\n")
    endif()
    
    # Build tool dependencies
    if(PKG_ROS_VERSION EQUAL 1)
        set(XML_CONTENT "${XML_CONTENT}  <buildtool_depend>catkin</buildtool_depend>\n")
    else()
        set(XML_CONTENT "${XML_CONTENT}  <buildtool_depend>ament_cmake</buildtool_depend>\n")
    endif()
    
    # Additional buildtool dependencies
    if(PKG_BUILDTOOL_DEPENDS)
        foreach(_dep ${PKG_BUILDTOOL_DEPENDS})
            set(XML_CONTENT "${XML_CONTENT}  <buildtool_depend>${_dep}</buildtool_depend>\n")
        endforeach()
    endif()
    
    set(XML_CONTENT "${XML_CONTENT}\n")
    
    # ROS package dependencies
    if(PKG_ROS_PACKAGES)
        # Convert to list if it's a string
        if(PKG_ROS_PACKAGES)
            string(REPLACE " " ";" PKG_ROS_PACKAGES "${PKG_ROS_PACKAGES}")
        endif()
        
        foreach(_pkg ${PKG_ROS_PACKAGES})
            if(PKG_ROS_VERSION EQUAL 1)
                # ROS 1 uses separate build_depend and exec_depend (or just depend for both)
                set(XML_CONTENT "${XML_CONTENT}  <depend>${_pkg}</depend>\n")
            else()
                # ROS 2 also uses depend for simplicity
                set(XML_CONTENT "${XML_CONTENT}  <depend>${_pkg}</depend>\n")
            endif()
        endforeach()
        set(XML_CONTENT "${XML_CONTENT}\n")
    endif()
    
    # Additional build dependencies
    if(PKG_BUILD_DEPENDS)
        foreach(_dep ${PKG_BUILD_DEPENDS})
            set(XML_CONTENT "${XML_CONTENT}  <build_depend>${_dep}</build_depend>\n")
        endforeach()
        set(XML_CONTENT "${XML_CONTENT}\n")
    endif()
    
    # Execution dependencies
    if(PKG_EXEC_DEPENDS)
        foreach(_dep ${PKG_EXEC_DEPENDS})
            set(XML_CONTENT "${XML_CONTENT}  <exec_depend>${_dep}</exec_depend>\n")
        endforeach()
        set(XML_CONTENT "${XML_CONTENT}\n")
    endif()
    
    # Test dependencies
    if(PKG_ROS_VERSION EQUAL 2)
        # ROS 2 common test dependencies
        set(XML_CONTENT "${XML_CONTENT}  <test_depend>ament_lint_auto</test_depend>\n")
        set(XML_CONTENT "${XML_CONTENT}  <test_depend>ament_lint_common</test_depend>\n")
    endif()
    
    if(PKG_TEST_DEPENDS)
        foreach(_dep ${PKG_TEST_DEPENDS})
            set(XML_CONTENT "${XML_CONTENT}  <test_depend>${_dep}</test_depend>\n")
        endforeach()
    endif()
    
    if(PKG_TEST_DEPENDS OR PKG_ROS_VERSION EQUAL 2)
        set(XML_CONTENT "${XML_CONTENT}\n")
    endif()
    
    # Documentation dependencies
    if(PKG_DOC_DEPENDS)
        foreach(_dep ${PKG_DOC_DEPENDS})
            set(XML_CONTENT "${XML_CONTENT}  <doc_depend>${_dep}</doc_depend>\n")
        endforeach()
        set(XML_CONTENT "${XML_CONTENT}\n")
    endif()
    
    # Export section
    set(XML_CONTENT "${XML_CONTENT}  <export>\n")
    if(PKG_ROS_VERSION EQUAL 2)
        set(XML_CONTENT "${XML_CONTENT}    <build_type>ament_cmake</build_type>\n")
    endif()
    set(XML_CONTENT "${XML_CONTENT}  </export>\n")
    
    # Close package tag
    set(XML_CONTENT "${XML_CONTENT}</package>\n")
    
    # Write the file
    file(WRITE "${PKG_OUTPUT_FILE}" "${XML_CONTENT}")
    
    message(STATUS "Generated package.xml for ROS ${PKG_ROS_VERSION}: ${PKG_OUTPUT_FILE}")
    message(STATUS "  Package: ${PKG_PACKAGE_NAME} v${PKG_VERSION}")
    message(STATUS "  ROS Packages: ${PKG_ROS_PACKAGES}")
endmacro()

# ============================================================================
# Function: print_package_xml_info
# ============================================================================
# Helper function to print information about the generated package.xml
function(print_package_xml_info PACKAGE_FILE)
    if(EXISTS "${PACKAGE_FILE}")
        message(STATUS "========================================")
        message(STATUS "Package XML Information:")
        message(STATUS "  File: ${PACKAGE_FILE}")
        message(STATUS "========================================")
    else()
        message(WARNING "Package file not found: ${PACKAGE_FILE}")
    endif()
endfunction()

# ============================================================================
# Function: validate_package_xml
# ============================================================================
# Validates that the generated package.xml contains required elements
function(validate_package_xml PACKAGE_FILE)
    if(NOT EXISTS "${PACKAGE_FILE}")
        message(FATAL_ERROR "Package file does not exist: ${PACKAGE_FILE}")
    endif()
    
    file(READ "${PACKAGE_FILE}" _content)
    
    # Check for required elements
    set(_required_elements "name" "version" "description" "maintainer" "license")
    foreach(_element ${_required_elements})
        string(FIND "${_content}" "<${_element}" _pos)
        if(_pos EQUAL -1)
            message(WARNING "Package XML missing required element: ${_element}")
        endif()
    endforeach()
    
    message(STATUS "Package XML validation complete: ${PACKAGE_FILE}")
endfunction()
