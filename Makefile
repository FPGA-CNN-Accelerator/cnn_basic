CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -ljpeg -lm

# Source files
SOURCES = main.c cnn.c
OBJECTS = $(SOURCES:.c=.o)
TARGET = cnn_mnist

# Default target
all: $(TARGET)

# Compile the executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install -y libjpeg-dev

# Run the program
run: $(TARGET)
	./$(TARGET)

# Debug build
debug: CFLAGS += -g -DDEBUG
debug: $(TARGET)

.PHONY: all clean install-deps run debug 