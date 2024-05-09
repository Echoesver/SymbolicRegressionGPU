# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/jiangg/mambaforge3/bin/cmake

# The command to remove a file.
RM = /home/jiangg/mambaforge3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build

# Include any dependencies generated for this target.
include CMakeFiles/cusr.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cusr.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cusr.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cusr.dir/flags.make

CMakeFiles/cusr.dir/src/prefix.cu.o: CMakeFiles/cusr.dir/flags.make
CMakeFiles/cusr.dir/src/prefix.cu.o: ../src/prefix.cu
CMakeFiles/cusr.dir/src/prefix.cu.o: CMakeFiles/cusr.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/cusr.dir/src/prefix.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cusr.dir/src/prefix.cu.o -MF CMakeFiles/cusr.dir/src/prefix.cu.o.d -x cu -dc /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/src/prefix.cu -o CMakeFiles/cusr.dir/src/prefix.cu.o

CMakeFiles/cusr.dir/src/prefix.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cusr.dir/src/prefix.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cusr.dir/src/prefix.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cusr.dir/src/prefix.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cusr.dir/src/regression.cu.o: CMakeFiles/cusr.dir/flags.make
CMakeFiles/cusr.dir/src/regression.cu.o: ../src/regression.cu
CMakeFiles/cusr.dir/src/regression.cu.o: CMakeFiles/cusr.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/cusr.dir/src/regression.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cusr.dir/src/regression.cu.o -MF CMakeFiles/cusr.dir/src/regression.cu.o.d -x cu -dc /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/src/regression.cu -o CMakeFiles/cusr.dir/src/regression.cu.o

CMakeFiles/cusr.dir/src/regression.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cusr.dir/src/regression.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cusr.dir/src/regression.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cusr.dir/src/regression.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cusr.dir/src/fit_eval.cu.o: CMakeFiles/cusr.dir/flags.make
CMakeFiles/cusr.dir/src/fit_eval.cu.o: ../src/fit_eval.cu
CMakeFiles/cusr.dir/src/fit_eval.cu.o: CMakeFiles/cusr.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/cusr.dir/src/fit_eval.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cusr.dir/src/fit_eval.cu.o -MF CMakeFiles/cusr.dir/src/fit_eval.cu.o.d -x cu -dc /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/src/fit_eval.cu -o CMakeFiles/cusr.dir/src/fit_eval.cu.o

CMakeFiles/cusr.dir/src/fit_eval.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cusr.dir/src/fit_eval.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cusr.dir/src/fit_eval.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cusr.dir/src/fit_eval.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cusr.dir/src/program.cu.o: CMakeFiles/cusr.dir/flags.make
CMakeFiles/cusr.dir/src/program.cu.o: ../src/program.cu
CMakeFiles/cusr.dir/src/program.cu.o: CMakeFiles/cusr.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/cusr.dir/src/program.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cusr.dir/src/program.cu.o -MF CMakeFiles/cusr.dir/src/program.cu.o.d -x cu -dc /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/src/program.cu -o CMakeFiles/cusr.dir/src/program.cu.o

CMakeFiles/cusr.dir/src/program.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cusr.dir/src/program.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cusr.dir/src/program.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cusr.dir/src/program.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/cusr.dir/run_cusr.cu.o: CMakeFiles/cusr.dir/flags.make
CMakeFiles/cusr.dir/run_cusr.cu.o: ../run_cusr.cu
CMakeFiles/cusr.dir/run_cusr.cu.o: CMakeFiles/cusr.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/cusr.dir/run_cusr.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/cusr.dir/run_cusr.cu.o -MF CMakeFiles/cusr.dir/run_cusr.cu.o.d -x cu -dc /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/run_cusr.cu -o CMakeFiles/cusr.dir/run_cusr.cu.o

CMakeFiles/cusr.dir/run_cusr.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cusr.dir/run_cusr.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/cusr.dir/run_cusr.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cusr.dir/run_cusr.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cusr
cusr_OBJECTS = \
"CMakeFiles/cusr.dir/src/prefix.cu.o" \
"CMakeFiles/cusr.dir/src/regression.cu.o" \
"CMakeFiles/cusr.dir/src/fit_eval.cu.o" \
"CMakeFiles/cusr.dir/src/program.cu.o" \
"CMakeFiles/cusr.dir/run_cusr.cu.o"

# External object files for target cusr
cusr_EXTERNAL_OBJECTS =

CMakeFiles/cusr.dir/cmake_device_link.o: CMakeFiles/cusr.dir/src/prefix.cu.o
CMakeFiles/cusr.dir/cmake_device_link.o: CMakeFiles/cusr.dir/src/regression.cu.o
CMakeFiles/cusr.dir/cmake_device_link.o: CMakeFiles/cusr.dir/src/fit_eval.cu.o
CMakeFiles/cusr.dir/cmake_device_link.o: CMakeFiles/cusr.dir/src/program.cu.o
CMakeFiles/cusr.dir/cmake_device_link.o: CMakeFiles/cusr.dir/run_cusr.cu.o
CMakeFiles/cusr.dir/cmake_device_link.o: CMakeFiles/cusr.dir/build.make
CMakeFiles/cusr.dir/cmake_device_link.o: CMakeFiles/cusr.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CUDA device code CMakeFiles/cusr.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cusr.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cusr.dir/build: CMakeFiles/cusr.dir/cmake_device_link.o
.PHONY : CMakeFiles/cusr.dir/build

# Object files for target cusr
cusr_OBJECTS = \
"CMakeFiles/cusr.dir/src/prefix.cu.o" \
"CMakeFiles/cusr.dir/src/regression.cu.o" \
"CMakeFiles/cusr.dir/src/fit_eval.cu.o" \
"CMakeFiles/cusr.dir/src/program.cu.o" \
"CMakeFiles/cusr.dir/run_cusr.cu.o"

# External object files for target cusr
cusr_EXTERNAL_OBJECTS =

cusr: CMakeFiles/cusr.dir/src/prefix.cu.o
cusr: CMakeFiles/cusr.dir/src/regression.cu.o
cusr: CMakeFiles/cusr.dir/src/fit_eval.cu.o
cusr: CMakeFiles/cusr.dir/src/program.cu.o
cusr: CMakeFiles/cusr.dir/run_cusr.cu.o
cusr: CMakeFiles/cusr.dir/build.make
cusr: CMakeFiles/cusr.dir/cmake_device_link.o
cusr: CMakeFiles/cusr.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CUDA executable cusr"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cusr.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cusr.dir/build: cusr
.PHONY : CMakeFiles/cusr.dir/build

CMakeFiles/cusr.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cusr.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cusr.dir/clean

CMakeFiles/cusr.dir/depend:
	cd /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2 /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2 /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build /home/jiangg/projects/machinelearning/TimeRelatedData/SymbolicRegressionGPUstaticv2/build/CMakeFiles/cusr.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cusr.dir/depend

