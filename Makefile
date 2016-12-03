# define the C compiler to use
CC = nvcc

# the build target executable
TARGET = main

# define project directory structure
SRCDIR = src
BINDIR = bin
TESTDIR = test

# define source files in project
SOURCES := $(wildcard $(SRCDIR)/*.cu)

# define include directory other than /usr/include
INCLUDE = -I./include

# define remove for cleaning
RM = rm -f

all: $(TARGET)

$(TARGET):
	$(CC) $(INCLUDE) -o $(BINDIR)/$(TARGET) $(SOURCES)

clean:
	$(RM) $(BINDIR)/$(TARGET)
