function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
n = size(theta);

hx = sigmoid(X*theta);
sum1 = sum(y .* log(hx) + (1 - y) .* log(1 - hx));
sum2 = sum(theta(2:n) .^ 2);
J = -1 / m * sum1 + lambda / (2 * m) * sum2;

grad(1) = 1 / m .* sum((hx - y) .* X(1));

for j = 2:n
  grad(j) = 1 / m .* sum((hx - y) .* X(:,j:j)) + lambda / m .* theta(j);
endfor




% =============================================================

end
