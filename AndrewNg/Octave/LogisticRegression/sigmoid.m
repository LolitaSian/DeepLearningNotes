function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
% 计算z的每个值的sigmoid（z可以是矩阵，向量或标量）。


g = 1 ./ ( 1 + exp( -1 * z ));

% =============================================================

end
