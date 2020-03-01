close all;
clear
clc
train_path = "pima-indians-diabetes.data_train.csv";
test_path = "pima-indians-diabetes.data_test.csv";
train_data = csvread(train_path);
test_data = csvread(test_path);
[train_rows, train_lists] = size(train_data);
train_label = train_data(:,train_lists);
train_data(:,train_lists) = [];
model = fitcnb(train_data,train_label);
[test_rows, test_lists] = size(test_data);
test_label = test_data(:,test_lists);
test_data(:,test_lists) = [];
predict_label = predict(model,test_data);
result = sum(predict_label == test_label)/test_rows;
result
