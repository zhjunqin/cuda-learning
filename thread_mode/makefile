#Makefile

SOURCE =  print_index.cu

CC = nvcc

exe: $(SOURCE)
        $(CC)    $(SOURCE)  -o print_index

clean:
        $(RM) -rf print_index

run:
        ./print_index
