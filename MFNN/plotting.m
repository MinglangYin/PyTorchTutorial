clear all
close all
clc

test = load('y_test.dat');
low = load('y_l.dat');
high = load('y_h.dat');

figure()
hold on
scatter(test(:,1), test(:,2), '.');
scatter(test(:,1), test(:,3), '.');
scatter(test(:,1), test(:,4), '.');

scatter(low(:,1), low(:,2), 'o');
scatter(high(:,1), high(:,2), 'x');

scatter(low(:,1), low(:,3), 'o');
scatter(high(:,1), high(:,3), 'x');