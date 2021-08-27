%(a)
tlin = linspace(0,1);
t = rand(30,1);
t = sort(t);
g = (sin(2*pi*t)).^2 + NormalDistribution(0,sqrt(0.07),t);
T_test = zeros(30,2);     T_test(:,1)=t(:,1); T_test(:,2)=g(:,1);
subplot(121)

plot(T_test(:,1),T_test(:,2),'o')
hold on

[B1,g1,MSE1] = PolyRegression(2,t,g,tlin);
plot(tlin,g1,'-r')
hold on

[B2,g2,MSE2] = PolyRegression(5,t,g,tlin);
plot(tlin,g2,'-g')
hold on

[B3,g3,MSE3] = PolyRegression(10,t,g,tlin);
plot(tlin,g3,'-b')
hold on

[B4,g4,MSE4] = PolyRegression(14,t,g,tlin);
plot(tlin,g4,'-black')
hold on

[B5,g5,MSE5] = PolyRegression(18,t,g,tlin);
plot(tlin,g5,'-m')
hold on
xlabel('time(s)')
ylabel('g(t)')
title('Overfitting Visualization')
legend('raw','2 orders','5 orders','10 orders','14 orders','18 orders');

%(b)
ekofS = zeros(18,1);
for i=2:18
    [B,G,MSE] = PolyRegression(i,t,g,tlin);
    ekofS(i) = MSE;
    
end
ekofS(1) = sum((t-g).^2);
ekofS = log(ekofS);

%(c)
t_test = rand(1000,1);
t_test = sort(t_test);
g_test = (sin(2*pi*t_test)).^2 + NormalDistribution(0,sqrt(0.07),t_test);
T_test = zeros(1000,2);     T_test(:,1) = t_test(:,1); T_test(:,2) = g_test(:,1);
ekofST = zeros(18,1);
for i=2:18
    [B,G,MSE] = PolyRegression(i,t,g,tlin);
    Gpredicted = B(1);
    for j=1:i-1
        if j==1
            continue
        end
        Gpredicted =  Gpredicted + t_test.^j*B(j+1);
    end
    for k=1:1000
       MSE = sum((Gpredicted - g_test).^2); 
    end
    ekofST(i) = MSE;
end
ekofST(1) = sum((t_test-g_test).^2);
ekofST = log(ekofST);

subplot(122)
plot(1:18,ekofS,1:18,ekofST)
xlabel('order')
ylabel('MSE(log)')
title('Training error v.s. Testing error')
legend('training error','testing error')

%functions
function [B,G,MSE] = PolyRegression(dim,t,g,tlin)
T = ones(length(t),1);
for i=1:dim
    T = [T t.^i];
end
B = T\g;
G = B(1);
Gpredicted = B(1);
for j=1:dim
    G = G + tlin.^j*B(j+1);
    Gpredicted =  Gpredicted + t.^j*B(j+1);
end
for k=1:length(g)
   MSE = sum((Gpredicted - g).^2); 
end
end

function N = NormalDistribution(miu,sigma,t)
N=(1/(sqrt(2*pi)*sigma))*exp(-0.5*((t-miu)/sigma).^2);
end