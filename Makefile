TARGET = $(addprefix bin/, $(basename $(wildcard *.cpp)))

all: $(TARGET)

CXX = g++
CXXFLAGS = -std=c++11 -g -Wall

LFLAGS = -Wall -I. -lboost_program_options

SOURCES := $(wildcard src/*.cpp)
OBJECTS := $(SOURCES:src/%.cpp=obj/%.o)

bin/%: %.cpp $(OBJECTS)
	@mkdir -p bin
	$(CXX) -o $@ $(LFLAGS) $(CXXFLAGS) $^

$(OBJECTS): obj/%.o : src/%.cpp src/%.h
	@mkdir -p obj
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONEY: clean
clean:
	rm -rf bin obj
