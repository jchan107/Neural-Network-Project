function showdigit(x,y)
% second argument is optional

d = sqrt(length(x));
x = reshape(x,[d d]);
image(x);
%image(255-x);
colormap gray;
if nargin>1
	title(sprintf('%d',y));
end;
