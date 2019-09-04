clear all
close all
clc

y_low = @(x) 0.5.*(6.*x-2).^2.*sin(12.*x-4)+10.*(x-0.5)-5;
y_high = @(x) (6.*x-2).^2.*sin(12.*x-4);
y_low_grad = @(x) sin(12.*x-4).*(6.*x-2).*6 + 6.*(6.*x-2).^2.*cos(12.*x-4) +10;
x = 0:0.01:1;

test = load('y_test.dat');
low = load('y_l.dat');
high = load('y_h.dat');
grad = load('y_l_grad.dat');

figure()
hold on
scatter(test(:,1), test(:,2), '.');
scatter(test(:,1), test(:,3), '.');
scatter(test(:,1), test(:,4), '.');

scatter(low(:,1), low(:,2), 'o');
scatter(low(:,1), low(:,3), 'o');

scatter(high(:,1), high(:,2), 'x');
scatter(high(:,1), high(:,3), 'x');

x_grad = grad(:,1);

scatter(grad(:,1), grad(:,2));
scatter(grad(:,1), grad(:,3));
% scatter(x_grad, y_low_grad(x_grad) );

grad(:,3)
y_low_grad(x_grad) 

plot(x, y_low(x))
plot(x, y_high(x))


% legend('yh pred test', 'y test real', 'yl pred test', 'yl pred', 'yl train', 'yh pred', 'yh train', 'yl grad pred', 'yl grad train', 'y_low_real', 'y_high_real')
% legend('yh pred test', 'y test real', 'yl pred test', 'yl pred', 'yl train', 'yh pred', 'yh train', 'y low real', 'y high real')




