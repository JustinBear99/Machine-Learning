%% load txt file and looking for the data we want, then plot the scatter plot
fid=fopen('02HW3_Diabold_Li_data.txt');
tline = fgetl(fid);
tlines = cell(0,1);
while ischar(tline)
    tlines{end+1,1} = tline;
    tline = fgetl(fid);
end
fclose(fid);

for i=1:length(tlines)
    if string(tlines(i)).count('19900531') == 1
        info(1) = string(tlines(i));
        info(2) = string(tlines(i+1));
    end
end
info = strcat(info(1),info(2));
info = info.split();
for i=2:(length(info))
    yield(i-1) = double(info(i));
end
 maturities = 1:(length(info)-1);
 scatter(maturities,yield)
 %% 1st order
x = linspace(0,18);
x1 = maturities';
Y = yield';
X1 = [ones(length(x1),1) x1];
B1 = X1\Y;
y1 = B1(1) + x*B1(2);
plot(x1,Y,'o')
hold on
plot(x,y1,'-r')
hold on
%% 2nd order
x2 = x1.^2;
X2 = [ones(length(x1),1) x1 x2];
B2 = X2\Y;
y2 = B2(1) + x*B2(2) + x.^2*B2(3);
plot(x,y2,'--g')
hold on
%% 3rd order
x3 = x1.^3;
X3 = [ones(length(x1),1) x1 x2 x3];
B3 = X3\Y;
y3 = B3(1) + x*B3(2) + x.^2*B3(3) + x.^3*B3(4);
plot(x,y3,'-.b')
%% 4th order
x4 = x1.^4;
X4 = [ones(length(x1),1) x1 x2 x3 x4];
B4 = X4\Y;
y4 = B4(1) + x*B4(2) + x.^2*B4(3) + x.^3*B4(4) + x.^4*B4(5);
plot(x,y4,'c')
%% 5th order
x5 = x1.^5;
X5 = [ones(length(x1),1) x1 x2 x3 x4 x5];
B5 = X5\Y;
y5 = B5(1) + x*B5(2) + x.^2*B5(3) + x.^3*B5(4) + x.^4*B5(5) + x.^5*B5(6);
plot(x,y5,'m')
%% 6th order
x6 = x1.^5;
X6 = [ones(length(x1),1) x1 x2 x3 x4 x5 x6];
B6 = X6\Y;
y6 = B6(1) + x*B6(2) + x.^2*B6(3) + x.^3*B6(4) + x.^4*B6(5) + x.^5*B6(6) + x.^6*B6(7);
plot(x,y6,'b')
legend('raw data', '1st order', '2nd order', '3rd order', '4th order', '5th order', '6th order')

%% calculate the residuals and plot scaater, histogram and qqplot
re_y1 = (B1(1) + x1*B1(2) - Y);
re_y2 = (B2(1) + x1*B2(2) + x1.^2*B2(3) - Y);
re_y3 = (B3(1) + x1*B3(2) + x1.^2*B3(3) + x1.^3*B3(4) - Y);
re_y4 = (B4(1) + x1*B4(2) + x1.^2*B4(3) + x1.^3*B4(4) + x1.^4*B4(5) - Y);
re_y5 = (B5(1) + x1*B5(2) + x1.^2*B5(3) + x1.^3*B5(4) + x1.^4*B5(5) + x1.^5*B5(6) - Y);
re_y6 = (B6(1) + x1*B6(2) + x1.^2*B6(3) + x1.^3*B6(4) + x1.^4*B6(5) + x1.^5*B6(6) + x1.^6*B6(7) - Y);
re_Y = [re_y1 re_y2 re_y3 re_y4 re_y5 re_y6];
disp(['The respective SSE are ',num2str(re_Y(1)),', ',num2str(re_Y(2)),', ',num2str(re_Y(3)),', ',num2str(re_Y(4)),', ',num2str(re_Y(5)),', ',num2str(re_Y(6))]);
subplot(131)
scatter(B6(1) + x1*B6(2) + x1.^2*B6(3) + x1.^3*B6(4) + x1.^4*B6(5) + x1.^5*B6(6) + x1.^6*B6(7),re_Y(:,6))
title('Residual plot of 6th order regression')
xlabel('fitted values')
ylabel('residuals')
subplot(132)
histogram(re_Y(:,6),10)
title('Histogram of 6th order regression')
xlabel('residual')
ylabel('numbers')
subplot(133)
qqplot(re_Y(:,6))
%% R square
xmean = mean(0:18);
ymean = mean(Y);
SSR_y1 = (B1(1) + x1*B1(2) - ymean).^2;
SSR_y2 = (B2(1) + x1*B2(2) + x1.^2*B2(3) - ymean).^2;
SSR_y3 = (B3(1) + x1*B3(2) + x1.^2*B3(3) + x1.^3*B3(4) - ymean).^2;
SSR_y4 = (B4(1) + x1*B4(2) + x1.^2*B4(3) + x1.^3*B4(4) + x1.^4*B4(5) - ymean).^2;
SSR_y5 = (B5(1) + x1*B5(2) + x1.^2*B5(3) + x1.^3*B5(4) + x1.^4*B5(5) + x1.^5*B5(6) - ymean).^2;
SSR_y6 = (B6(1) + x1*B6(2) + x1.^2*B6(3) + x1.^3*B6(4) + x1.^4*B6(5) + x1.^5*B6(6) + x1.^6*B6(7) - ymean).^2;
SSR = [sum(SSR_y1)/(sum(re_y1)+sum(SSR_y1)) sum(SSR_y2)/(sum(re_y2)+sum(SSR_y2)) sum(SSR_y3)/(sum(re_y3)+sum(SSR_y3)) sum(SSR_y4)/(sum(re_y4)+sum(SSR_y4)) sum(SSR_y5)/(sum(re_y5)+sum(SSR_y5)) sum(SSR_y6)/(sum(re_y6)+sum(SSR_y6)) ];
scatter(1:6,SSR)
hold on
plot(1:6,smooth(1:6,SSR,'rloess'))
ylim([0 1])
%% 3. c)
scatter(1:18,(B4(1) + x1*B4(2) + x1.^2*B4(3) + x1.^3*B4(4) + x1.^4*B4(5) - Y))
hold on
plot(0:18,[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ]')
title('residual plot (vs maturity)')
xlabel('maturities')
ylabel('residual')