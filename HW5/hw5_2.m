fid=fopen('05HW2_wine_training.txt');
tline = fgetl(fid);
tlines = cell(0,1);
while ischar(tline)
    tlines{end+1,1} = tline;
    tline = fgetl(fid);
end
fclose(fid);
wine_training = zeros(length(tlines),12);
for i = 1:length(tlines)
    wine_training(i,1:12) = string(tlines(i)).split();
end

fid=fopen('05HW2_wine_test.txt');
tline = fgetl(fid);
tlines = cell(0,1);
while ischar(tline)
    tlines{end+1,1} = tline;
    tline = fgetl(fid);
end
fclose(fid);
wine_test = zeros(length(tlines),12);
for i = 1:length(tlines)
    wine_test(i,1:12) = string(tlines(i)).split();
end
%%
my_lambda = [0.0001 0.0005 0.0025 0.0125 0.0625 0.3125 1.5625 7.815 39.0625 195.3125];
fit = glmnet(wine_training(:,1:11), wine_training(:,12));
for i = 1:length(my_lambda)
    beta(:,i) = glmnetCoef(fit, my_lambda(i));
    pred(:,i) = glmnetPredict(fit, wine_test(:,1:11), my_lambda(i), 'link');
end
for i = 1:10
    y_train(:,i) = [ones(100,1) wine_training(:,1:11)]*beta(:,i);
end
train_error(:,:) = (y_train(:,:) - wine_training(:,12)).^2;
train_error(101,:) = sum(train_error(1:100,:));
test_error(:,:) = (pred(:,:) - wine_test(:,12)).^2;
test_error(101,:) = sum(test_error(1:100,:));
scatter(log(my_lambda), log(train_error(101,:)))
hold on
scatter(log(my_lambda), log(test_error(101,:)))
title('Square error of lasso approach')
xlabel('log of \lambda')
ylabel('log of square error')
legend('train error', 'test error')