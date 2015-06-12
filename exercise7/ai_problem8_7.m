
clear
x = 0;
y = 0;
theta = 0;


w = pi/8;
delta_t=4;
v = 10;



for i = 1:4
	x = x + v*cos(theta)*delta_t
	y = y+ v*sin(theta)*delta_t
	theta = theta + w*delta_t 
	disp('...........')
end

theta == w*16
x
y
theta