objects = DigitReader.o mnistloader.o
libs = -lboost_iostreams -lfann
opts = -Wall -std=c++11 -g
cc = g++ $(opts)

DigitReader: $(objects)
	$(cc) $^ -o $@ $(libs)

%.o: %.cpp
	$(cc) -c $< -o $@

clean:
	rm -f $(objects)
.PHONY: clean
