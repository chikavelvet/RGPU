CPPC=g++
CUDA=nvcc
CFLAGS=-g -std=c++0x -I.
CUDAFLAGS=-g -std=c++11 -arch=sm_61 -I.

SRCDIR=./src
BINDIR=./bin
OBJDIR=./obj

_FILES=main

_OBJ=$(patsubst %,%.o,$(_FILES))
_SRC=$(patsubst %,%.cpp,$(_FILES))
_CUDASRC=$(patsubst %,%.cu,$(_FILES))

OBJ=$(patsubst %,$(OBJDIR)/%,$(_OBJ))
SRC=$(patsubst %,$(SRCDIR)/%,$(_SRC))
CUDASRC=$(patsubst %,$(SRCDIR)/%,$(_CUDASRC))

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CPPC) -c -o $@ $(CFLAGS)

rgpu: $(SRC)
	$(CPPC) -o $(BINDIR)/$@ $^ $(CFLAGS)

test: ./bin/rgpu ./testinput.txt
	./bin/rgpu --input ./testinput.txt --debug

.PHONY: clean

clean:
	rm -f $(OBJDIR)/*.o *~ core $(BINDIR)/*

