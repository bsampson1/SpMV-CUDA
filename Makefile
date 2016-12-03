# define the C compiler to use
CC = nvcc

# define project directory structure
SRCDIR = src
BINDIR = bin
TESTDIR = test

# the build target executable
TARGET = $(BINDIR)/main

# define source files in project
SOURCES := $(wildcard $(SRCDIR)/*.cu)

# define include directory other than /usr/include
INCLUDE = -I./include

# define remove for cleaning
RM = rm -f

all: $(TARGET)

$(TARGET):
	$(CC) $(INCLUDE) -o $(TARGET) $(SOURCES)

run:
	./$(TARGET)

clean:
	$(RM) $(TARGET)
