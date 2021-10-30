

function f
  options = optimset ('GradObj','on','MaxIter','100');
  initheta = zeros(2,1);
  [Theta,J,Flag] = fminunc (@costFunction,initheta,options)
end

function J = cost_function(X,y,theta)
  m = length(y);                % ���m
  predict = X*theta;            % ����X*theta
  result = (predict - y) .^ 2;  % ����֮��ƽ��
  J = 1/(2*m) * sum(result);    % �Խ����Ͳ���2m
endfunction