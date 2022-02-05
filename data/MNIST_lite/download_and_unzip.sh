rm -rf mnist
rm -rf train
rm -rf test

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz

mkdir train
mkdir test

mv train-images-idx3-ubyte train
mv train-labels-idx1-ubyte train
mv t10k-images-idx3-ubyte test
mv t10k-labels-idx1-ubyte tests

rm -rf mnist
rm -rf MNIST.zip