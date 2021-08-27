xlin = linspace(0,2);
x = [0.2;0.3;0.6;0.9;1.1;1.3;1.4;1.6];
y = [0.050446;0.098426;0.33277;0.7266;1.0972;1.5697;1.8487;2.5015];
[B1,Y1,MSE1] = PolyRegression(1,x,y,xlin);
[B2,Y2,MSE2] = PolyRegression(2,x,y,xlin);
[B3,Y3,MSE3] = PolyRegression(3,x,y,xlin);
subplot(121)
plot(x,y,'o')
hold on
plot(xlin,Y1,'-r',xlin,Y2,'-g',xlin,Y3,'-b')
xlabel('x')
ylabel('y')
title('Polynomial models over given data')
legend('raw','1 order','2 orders','3 orders')

RSS = [MSE1;MSE2;MSE3];
for i=1:3
    AIC(i)=length(y)*log(RSS(i)/length(y))+2*i;
    BIC(i)=length(y)*log(RSS(i)/length(y))+log(length(y))*i;
end
subplot(122)
bar([AIC(1:3);BIC(1:3)]', 1);
title('AIC and BIC');
xlabel('Order');
ylabel('AIC/BIC');
legend('AIC', 'BIC');

function [B,Y,MSE] = PolyRegression(dim,x,y,tlin)
X = ones(length(x),1);
for i=1:dim
    X = [X x.^i];
end
B = X\y;
Y = B(1);
Ypredicted = B(1);
for j=1:dim
    Y = Y + tlin.^j*B(j+1);
    Ypredicted =  Ypredicted + x.^j*B(j+1);
end
for k=1:length(y)
   MSE = sum((Ypredicted - y).^2); 
end
end