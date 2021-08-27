X = 0:0.01:2;
Y = sin(2*pi*X)+.1*rand(size(X));
net = newff([0 2],[3 1],{'tansig' 'purelin'});
net.trainParam.epochs = 2000;
net.trainParam.goal = 0.001;
net = train( net, X, Y);
simY = sim(net, X);
RMSE = sqrt(sum((simY(:)-Y(:)).^2)/length(Y));


%%

data = load('07HW2_digit.mat');
train_x = [reshape(data.train0,[500,28,28]); reshape(data.train1,[500,28,28]); reshape(data.train2,[500,28,28]); reshape(data.train3,[500,28,28]);
           reshape(data.train4,[500,28,28]); reshape(data.train5,[500,28,28]); reshape(data.train6,[500,28,28]); reshape(data.train7,[500,28,28]);
           reshape(data.train8,[500,28,28]); reshape(data.train9,[500,28,28])];
train_x = permute(train_x,[2,3,1]);
label = eye(10);
train_y = [repmat(label(1,:),500,1); repmat(label(2,:),500,1); repmat(label(3,:),500,1); repmat(label(4,:),500,1);
           repmat(label(5,:),500,1); repmat(label(6,:),500,1); repmat(label(7,:),500,1); repmat(label(8,:),500,1);
           repmat(label(9,:),500,1); repmat(label(10,:),500,1)];
train_y = permute(train_y,[2,3,1]);
      
cnn.layers = {
    struct('type', 'i')                                     %input layer
    struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5)   %convolution layer
    struct('type', 's', 'scale', 2)                         %sub sampling layer
    struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5)  %convolution layer
    struct('type', 's', 'scale', 2)                         %subsampling layer
};
 
opts.alpha = 2;
opts.batchsize = 50;
opts.numepochs = 10;

rand('state',0);
cnn = cnnsetup(cnn, train_x, train_y);
cnn = cnntrain(cnn, train_x, train_y, opts);
%%
test_x = [reshape(data.test0,[100,28,28]); reshape(data.test1,[100,28,28]); reshape(data.test2,[100,28,28]); reshape(data.test3,[100,28,28]);
          reshape(data.test4,[100,28,28]); reshape(data.test5,[100,28,28]); reshape(data.test6,[100,28,28]); reshape(data.test7,[100,28,28]);
          reshape(data.test8,[100,28,28]); reshape(data.test9,[100,28,28])];
test_x = permute(test_x,[2,3,1]);
test_y = [repmat(label(1,:),100,1); repmat(label(2,:),100,1); repmat(label(3,:),100,1); repmat(label(4,:),100,1);
           repmat(label(5,:),100,1); repmat(label(6,:),100,1); repmat(label(7,:),100,1); repmat(label(8,:),100,1);
           repmat(label(9,:),100,1); repmat(label(10,:),100,1)];
test_y = permute(test_y,[2,3,1]); 

[er, bad] = cnntest(cnn, test_x, test_y);
%%
figure;
plot(cnn.rL);
title('Mean square error')
xlabel('epochs')
xticklabels([0 1 2 3 4 5 6 7 8 9 10])
ylabel('error (percentage)')
%%
figure;
plot(rl_001_50_10)
hold on
plot(rl_01_50_10)
hold on
plot(rl_1_50_10)
hold on
plot(rl_15_50_10)
hold on
plot(cnn.rL)
legend('\alpha=0.01 bs=50 epochs=10','\alpha=0.1 bs=50 epochs=10', '\alpha=1 bs=50 epochs=10', '\alpha=1.5 bs=50 epochs=10', '\alpha=2 bs=50 epochs=10')
xlabel('iterations')
ylabel('mean square error')
title('When \alpha changes')
%%
figure;
plot(rl_1_50_10)
hold on
plot(rl_1_100_10)
hold on
plot(rl_1_200_10)
legend('\alpha=1 bs=50 epochs=10','\alpha=1 bs=100 epochs=10', '\alpha=1 bs=200 epochs=10')
xlabel('iterations')
ylabel('mean square error')
title('When batchsizes changes')
%%
figure;
plot(rl_1_50_10)
hold on
plot(rl_1_50_30)
hold on
plot(rl_1_50_50)
legend('\alpha=1 bs=50 epochs=10','\alpha=1 bs=50 epochs=30', '\alpha=1 bs=50 epochs=50')
xlabel('iterations')
ylabel('mean square error')
title('When epochs changes')