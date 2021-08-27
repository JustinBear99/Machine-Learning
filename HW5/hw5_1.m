load 05HW1_diabetes.mat;
my_lambda = [0.01 0.1 1 10 100 1000];
fit = glmnet(x_train, y_train);
for i = 1:length(my_lambda)
    beta(:,i) = glmnetCoef(fit, my_lambda(i));
    pred(:,i) = glmnetPredict(fit, x_test, my_lambda(i), 'link');
end
%%
% a)
subplot(121)
plot(1:65,beta(:,1))
hold on
plot(1:65,beta(:,2))
hold on
plot(1:65,beta(:,3))
hold on
plot(1:65,beta(:,4))
hold on
plot(1:65,beta(:,5))
hold on
plot(1:65,beta(:,6))
hold on
legend('\lambda = 0.01', '\lambda = 0.1', '\lambda = 1', '\lambda = 10', '\lambda = 100', '\lambda = 1000')
title("LASSO regression's \beta changes under different \lambda")
xlabel('coefficients')
ylabel('\beta')
subplot(122)
plot(1:6,beta(2:65,:))
title("LASSO regression's coefficients shrinkage under different \lambda")
xlabel('\lambda')
ylabel('coefficients')
xticks([1 2 3 4 5 6])
xticklabels({'0.01', '0.1', '1', '10', '100', '1000'})
%%
% b)
test_error(:,:) = (pred(:,:) - y_test(:,1)).^2;
test_error(201,:) = sum(test_error(1:200,:));
scatter(log(my_lambda), log(test_error(201,:)))
title('Square error of lasso approach')
xlabel('log of \lambda')
ylabel('log of square error')
%%
% c)
subplot(121)
B = ridge(y_train,x_train,my_lambda);
plot(1:64,B(:,1))
hold on
plot(1:64,B(:,2))
hold on
plot(1:64,B(:,3))
hold on
plot(1:64,B(:,4))
hold on
plot(1:64,B(:,5))
hold on
plot(1:64,B(:,6))
hold on
legend('\lambda = 0.01', '\lambda = 0.1', '\lambda = 1', '\lambda = 10', '\lambda = 100', '\lambda = 1000')
title("Ridge regression's \beta changes under different \lambda")
xlabel('coefficients')
ylabel('\beta')
subplot(122)
plot(1:6,B(1:64,:))
title("Ridge regression's coefficients shrinkage under different \lambda")
xlabel('\lambda')
ylabel('coefficients')
xticks([1 2 3 4 5 6])
xticklabels({'0.01', '0.1', '1', '10', '100', '1000'})
%%
y_ridge = x_test*B;
test_error(:,:) = (y_ridge(:,:) - y_test(:,1)).^2;
test_error(201,:) = sum(test_error(1:200,:));
scatter(log(my_lambda), log(test_error(201,:)))
title('Square error of ridge approach')
xlabel('log of \lambda')
ylabel('log of square error')