# ppi_experiment

1. clone the repo
2. unzip the zip file(uncompressed file is 1.8 gb)
3. run train_nn.py

# v 2
1. clone the repo
2. unzip the zip file
3. shuffle the file:
  shuf data_15000.csv > data_15000_shuf.csv
4. split train and test:
  head -n 24000 data_15000_shuf.csv > train.csv
  tail -n 6000 data_15000_shuf.csv > test.csv
dependencies:
  numpy, tensorflow
