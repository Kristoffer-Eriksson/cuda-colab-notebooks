# cuda-raytrace

A raytracer in cuda that can run in a colab notebook

## Run the POC in colab

!git clone https://github.com/Kristoffer-Eriksson/cuda-raytrace.git

!nvcc cuda-raytrace/src/poc.cu -o poc

!nvprof ./poc

