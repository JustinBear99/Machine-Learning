%% 1st order regression
x = linspace(0,1);
x1 = load('02HW1_Xtrain');
Y = load('02HW1_Ytrain ');
X1 = [ones(length(x1),1) x1];
B1 = X1\Y;
y1 = B1(1) + x*B1(2);
plot(x1,Y,'o')
hold on
plot(x,y1,'-r')
hold on
%% 2nd order regression
x2 = x1.^2;
X2 = [ones(length(x1),1) x1 x2];
B2 = X2\Y;
y2 = B2(1) + x*B2(2) + x.^2*B2(3);
plot(x,y2,'--g')
hold on
%% 3rd order regression
x3 = x1.^3;
X3 = [ones(length(x1),1) x1 x2 x3];
B3 = X3\Y;
y3 = B3(1) + x*B3(2) + x.^2*B3(3) + x.^3*B3(4);
plot(x,y3,'-.b')
legend('raw data', '1st order', '2nd order', '3rd order')
%% residual calculation
residual_y1 = (B1(1) + x1*B1(2) - Y);
residual_y2 = (B2(1) + x1*B2(2) + x1.^2*B2(3) - Y);
residual_y3 = (B3(1) + x1*B3(2) + x1.^2*B3(3) +x1.^3*B3(4) - Y);
residual_Y = sum([(residual_y1).^2 (residual_y2).^2 (residual_y3).^2]);
disp(['The respective SSE are ',num2str(residual_Y(1)),', ',num2str(residual_Y(2)),', ',num2str(residual_Y(3))]);
%% residual analysis
% linearity
subplot(221)
plot(x1,residual_y1,'or')
hold on
plot(x1,residual_y2,'og')
hold on
plot(x1,residual_y3,'ob')
hold on
legend('1st order','2nd order','3rd order')
title('Linearity')
xlabel('xtrain')
ylabel('residual')

%Homoscedasticity
subplot(222)
residuals_y1 = zeros(20,1);
residuals_y2 = zeros(20,1);
residuals_y3 = zeros(20,1);
for i = 1:20
    residuals_y1(i) = residual_y1(i)/x1(i);
    residuals_y2(i) = residual_y2(i)/x1(i);
    residuals_y3(i) = residual_y3(i)/x1(i);
end
plot(x1,residuals_y1,'or')
hold on
plot(x1,residuals_y2,'og')
hold on
plot(x1,residuals_y3,'ob')
hold on
legend('1st order','2nd order','3rd order')
title('Homoscedasticity')
xlabel('xtrain')
ylabel('residual / xtrain')

%Normal distribution
subplot(223)
histogram(residual_y1)
hold on
histogram(residual_y2)
hold on
histogram(residual_y3)
legend('1st order','2nd order','3rd order')
title('Normal distribution')
xlabel('residual')
ylabel('numbers')

%independence
subplot(224)
plot(B1(1) + x1*B1(2),residual_y1,'or')
hold on
plot(B2(1) + x1*B2(2) + x1.^2*B2(3),residual_y2,'og')
hold on
plot(B3(1) + x1*B3(2) + x1.^2*B3(3) +x1.^3*B3(4),residual_y3,'ob')
hold on
legend('1st order','2nd order','3rd order')
title('Independence')
xlabel('fitted values')
ylabel('residual')
%%
% subplot(221)
% plot(B1(1) + x1*B1(2),residual_y1,'or')
% title('Linearity')
% subplot(222)
% plot(x1,residuals_y1,'or')
% title('Homoscedasticity')
% subplot(223)
% histogram(residual_y1)
% title('Normal distribution')
% subplot(224)
% plot(x1,residual_y1,'or')
% title('Independence')
% %%
% subplot(221)
% plot(B2(1) + x1*B2(2) + x1.^2*B2(3),residual_y2,'og')
% title('Linearity')
% subplot(222)
% plot(x1,residuals_y2,'og')
% title('Homoscedasticity')
% subplot(223)
% histogram(residual_y2)
% title('Normal distribution')
% subplot(224)
% plot(x1,residual_y2,'og')
% title('Independence')
% %%
% subplot(221)
% plot(B3(1) + x1*B3(2) + x1.^2*B3(3) +x1.^3*B3(4),residual_y3,'ob')
% title('Linearity')
% subplot(222)
% plot(x1,residuals_y3,'ob')
% title('Homoscedasticity')
% subplot(223)
% histogram(residual_y3,10)
% title('Normal distribution')
% subplot(224)
% plot(x1,residual_y3,'ob')
% title('Independence')