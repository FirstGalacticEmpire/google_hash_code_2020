# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2020.2\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2020.2\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\Janek_PC\Desktop\Projects\uopo2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\Janek_PC\Desktop\Projects\uopo2\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\uopo2.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\uopo2.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\uopo2.dir\flags.make

CMakeFiles\uopo2.dir\main.cpp.obj: CMakeFiles\uopo2.dir\flags.make
CMakeFiles\uopo2.dir\main.cpp.obj: ..\main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Janek_PC\Desktop\Projects\uopo2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/uopo2.dir/main.cpp.obj"
	C:\PROGRA~2\MICROS~2\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\uopo2.dir\main.cpp.obj /FdCMakeFiles\uopo2.dir\ /FS -c C:\Users\Janek_PC\Desktop\Projects\uopo2\main.cpp
<<

CMakeFiles\uopo2.dir\main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/uopo2.dir/main.cpp.i"
	C:\PROGRA~2\MICROS~2\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx86\x86\cl.exe > CMakeFiles\uopo2.dir\main.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\Janek_PC\Desktop\Projects\uopo2\main.cpp
<<

CMakeFiles\uopo2.dir\main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/uopo2.dir/main.cpp.s"
	C:\PROGRA~2\MICROS~2\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\uopo2.dir\main.cpp.s /c C:\Users\Janek_PC\Desktop\Projects\uopo2\main.cpp
<<

CMakeFiles\uopo2.dir\Library.cpp.obj: CMakeFiles\uopo2.dir\flags.make
CMakeFiles\uopo2.dir\Library.cpp.obj: ..\Library.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Janek_PC\Desktop\Projects\uopo2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/uopo2.dir/Library.cpp.obj"
	C:\PROGRA~2\MICROS~2\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\uopo2.dir\Library.cpp.obj /FdCMakeFiles\uopo2.dir\ /FS -c C:\Users\Janek_PC\Desktop\Projects\uopo2\Library.cpp
<<

CMakeFiles\uopo2.dir\Library.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/uopo2.dir/Library.cpp.i"
	C:\PROGRA~2\MICROS~2\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx86\x86\cl.exe > CMakeFiles\uopo2.dir\Library.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\Janek_PC\Desktop\Projects\uopo2\Library.cpp
<<

CMakeFiles\uopo2.dir\Library.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/uopo2.dir/Library.cpp.s"
	C:\PROGRA~2\MICROS~2\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx86\x86\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\uopo2.dir\Library.cpp.s /c C:\Users\Janek_PC\Desktop\Projects\uopo2\Library.cpp
<<

# Object files for target uopo2
uopo2_OBJECTS = \
"CMakeFiles\uopo2.dir\main.cpp.obj" \
"CMakeFiles\uopo2.dir\Library.cpp.obj"

# External object files for target uopo2
uopo2_EXTERNAL_OBJECTS =

uopo2.exe: CMakeFiles\uopo2.dir\main.cpp.obj
uopo2.exe: CMakeFiles\uopo2.dir\Library.cpp.obj
uopo2.exe: CMakeFiles\uopo2.dir\build.make
uopo2.exe: CMakeFiles\uopo2.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\Janek_PC\Desktop\Projects\uopo2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable uopo2.exe"
	"C:\Program Files\JetBrains\CLion 2020.2\bin\cmake\win\bin\cmake.exe" -E vs_link_exe --intdir=CMakeFiles\uopo2.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x86\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100183~1.0\x86\mt.exe --manifests  -- C:\PROGRA~2\MICROS~2\2019\COMMUN~1\VC\Tools\MSVC\1427~1.291\bin\Hostx86\x86\link.exe /nologo @CMakeFiles\uopo2.dir\objects1.rsp @<<
 /out:uopo2.exe /implib:uopo2.lib /pdb:C:\Users\Janek_PC\Desktop\Projects\uopo2\cmake-build-debug\uopo2.pdb /version:0.0  /machine:X86 /debug /INCREMENTAL /subsystem:console  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib 
<<

# Rule to build all files generated by this target.
CMakeFiles\uopo2.dir\build: uopo2.exe

.PHONY : CMakeFiles\uopo2.dir\build

CMakeFiles\uopo2.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\uopo2.dir\cmake_clean.cmake
.PHONY : CMakeFiles\uopo2.dir\clean

CMakeFiles\uopo2.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\Janek_PC\Desktop\Projects\uopo2 C:\Users\Janek_PC\Desktop\Projects\uopo2 C:\Users\Janek_PC\Desktop\Projects\uopo2\cmake-build-debug C:\Users\Janek_PC\Desktop\Projects\uopo2\cmake-build-debug C:\Users\Janek_PC\Desktop\Projects\uopo2\cmake-build-debug\CMakeFiles\uopo2.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\uopo2.dir\depend
