%% Problem2
%(a)
X = [1 1;3 3;4 2];
y = [1;2;3];
classifier = fitcknn(X,y);
x1range = min(X(:,1)):.01:max(X(:,1));
x2range = min(X(:,2)):.01:max(X(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];
predictedspecies = predict(classifier,XGrid);
subplot(121)
gscatter(xx1(:), xx2(:), predictedspecies,'rgb');
hold on
scatter(X(:,1), X(:,2));
%(b)
w = [0.5;1];
dM = @(X1,X2)sqrt((bsxfun(@minus,X1,X2).^2)*w);
classifier2 = fitcknn(X,y,'Distance',dM,'NumNeighbors',1);
predictedspecies2 = predict(classifier2, XGrid);
subplot(122)
gscatter(xx1(:), xx2(:), predictedspecies2,'rgb');
hold on
scatter(X(:,1), X(:,2));

%% Problem3
%(a)
fid=fopen('10HW3_train.txt');
A = fscanf(fid,'%f');
train = reshape(A,[785,1000])';
fid=fopen('10HW3_test.txt');
A = fscanf(fid,'%f');
test = reshape(A,[785,300])';
fid=fopen('10HW3_validate.txt');
A = fscanf(fid,'%f');
validate = reshape(A,[785,300])';
error = zeros(6,2);
test_result = zeros(300,6);
val_result = zeros(300,6);
%%
for i = 1:6
    kk = [1 3 5 11 16 21];
    classifier{i} = fitcknn(train(:,1:784),train(:,785),'NumNeighbors',kk(i));
    test_result(:,i) = predict(classifier{i},test(:,1:784));
    val_result(:,i) = predict(classifier{i}, validate(:,1:784));
    for j = 1:300
        if test(j,785) ~= test_result(j,i)
           error(i,1) = error(i,1) + 1;
        end
       if validate(j,785) ~= val_result(j,i)
           error(i,2) = error(i,2) + 1;
       end
    end
end
%%
figure;
plot(1:6,error(:,1),'ro-')
hold on
plot(1:6,error(:,2),'bo-')
xticks([1,2,3,4,5,6])
xticklabels([1,3,5,11,16,21])
xlabel('value of k')
ylabel('error (#)')
title('Test and validation error upon different value of k')
legend({'test error','validate error'})
%%
%(b)
g1 = test(:,785);
g2 = test_result(:,2);
C = confusionmat(g1,g2);
cm = confusionchart(C);
%%
%(c)
config = [];
for i=1:300
    if (test(i,785) == 3 && test_result(i,2) ~= 3) || (test(i,785) ~= 3 && test_result(i,2) == 3)
        config = [config i];
    end
end
for j = 1:length(config)
    subplot(1,length(config),j)
    imshow(reshape(test(config(j),1:784),[28,28])')
end