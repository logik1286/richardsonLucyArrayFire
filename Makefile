CC=g++
CFLAGS=-std=c++11
OBJ= richardsonLucy.o
LIBS=-laf -lafopencl -lafcpu
EXE=rlaf
all: rlaf

%.o : %.cpp
	g++ $(CFLAGS) -c $< -o $@


$(EXE): $(OBJ)
	g++ $(CFLAGS) $< $(LIBS) -o $(EXE)

clean:
	rm *.o $(EXE)
