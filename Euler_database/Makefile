# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -Wall -Wextra -O3

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Executable targets
TARGETS = $(BIN_DIR)/input_generator $(BIN_DIR)/run_euler_solver $(BIN_DIR)/create_csv_database

# Source files
SRCS = input_generator.cpp run_euler_solver.cpp create_csv_database.cpp

# Object files (stored in the build directory)
OBJS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(notdir $(SRCS)))

# Default target (build all executables)
all: $(TARGETS)

# Rule to create the build and bin directories if they don't exist
$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

# Rule to build input_generator
$(BIN_DIR)/input_generator: $(BUILD_DIR)/input_generator.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Rule to build run_euler_solver
$(BIN_DIR)/run_euler_solver: $(BUILD_DIR)/run_euler_solver.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Rule to build create_csv_database
$(BIN_DIR)/create_csv_database: $(BUILD_DIR)/create_csv_database.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $<

# Rule to compile source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Phony targets (not actual files)
.PHONY: all clean
