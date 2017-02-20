addpath('/home/fu/workspace/deeplearning/caffe/matlab');

caffe.reset_all;
caffe.set_mode_gpu();
caffe.set_device(0);

solver = caffe.Solver('lstm_bits_solver.prototxt');
solver.step(1);

data = solver.net.blobs('data').get_data();
label = solver.net.blobs('label').get_data();
cont = solver.net.blobs('cont').get_data();