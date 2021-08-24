

function f
  options = optimset ('GradObj','on','MaxIter','100');
  initheta = zeros(2,1);
  [Theta,J,Flag] = fminunc (@costFunction,initheta,options)
end

function J = cost_function(X,y,theta)
  m = length(y);                % 获得m
  predict = X*theta;            % 计算X*theta
  result = (predict - y) .^ 2;  % 减法之后平方
  J = 1/(2*m) * sum(result);    % 对结果求和并÷2m
endfunction