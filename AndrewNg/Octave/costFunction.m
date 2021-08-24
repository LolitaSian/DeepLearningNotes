function [j,gradient] = costFunction(theta)
  %代价函数J
  j = (theta(1)-5)^2 + (theta(2)-5)^2;
  
  gradient = zeros(2,1);
  
  %偏导
  gradient(1) = 2*(theta(1)-5);
  gradient(2) = 2*(theta(2)-5);
  
 endfunction