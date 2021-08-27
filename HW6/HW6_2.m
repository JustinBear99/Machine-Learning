%a
load iris.mat;
N = normalize(Inputs);
[U,S,V] = svd(N);
totalvar = S(1,1)+S(2,2)+S(3,3)+S(4,4);
accumuvar = [S(1,1)/totalvar (S(1,1)+S(2,2))/totalvar (S(1,1)+S(2,2)+S(3,3))/totalvar 1];
new_S = S;
new_S(3,3) = 0;
new_S(4,4) = 0;
new_Inputs = U(1:2,:)*new_S*V;
figure
scatter(new_Inputs(1,1:50),new_Inputs(2,1:50),'r')
hold on
scatter(new_Inputs(1,51:100),new_Inputs(2,51:100),'g')
hold on
scatter(new_Inputs(1,101:150),new_Inputs(2,101:150),'b')
title('Normalized data points on first 2 PCs')
xlabel('PC1')
ylabel('PC2')
legend('setosa','versicolor','virginica')
%%
%b(1)
hw6_2(N)

%%
%b(2)
hw6_2(new_Inputs)

function hw6_2(N)
training_data = [N(:,1:30) N(:,51:80) N(:,101:130)];
training_label = [ones(1,30) ones(1,30)*2 ones(1,30)*3];
testing_data = [N(:,31:50) N(:,81:100) N(:,131:150)];
testing_label = [ones(1,20) ones(1,20)*2 ones(1,20)*3];
[v,d] = LDA(training_data,training_label);

new_d = [d(:,1) d(:,2)];

training_data_lda = training_data'*new_d;
testing_data_lda = testing_data'*new_d;
median_lda_y = [median(training_data_lda(1:30,2)) median(training_data_lda(31:60,2)) median(training_data_lda(61:90,2))];
classifier = [(median_lda_y(1)+median_lda_y(2))/2 (median_lda_y(2)+median_lda_y(3))/2];

subplot(121)
scatter(training_data_lda(1:30,1),training_data_lda(1:30,2),'r')
hold on
scatter(training_data_lda(31:60,1),training_data_lda(31:60,2),'g')
hold on
scatter(training_data_lda(61:90,1),training_data_lda(61:90,2),'b')
hold on
x = linspace(min(training_data_lda(:,1)*1.1),max(training_data_lda(:,1)*1.1));
plot(x,classifier(1)+x*0,'black')
hold on
plot(x,classifier(2)+x*0,'black')
title('Training via PCA & LDA')
xlabel('LD1')
ylabel('LD2')
legend('setosa','versicolor','virginica')

training_error=0;
for i=1:30
    if training_data_lda(i,2) < classifier(1)
        training_error = training_error + 1;
    end
    if training_data_lda(i+30,2) > classifier(1) || training_data_lda(i+30,2) < classifier(2)
        training_error = training_error + 1;
    end
    if training_data_lda(i+60,2) > classifier(2)
        training_error = training_error + 1;
    end
end

subplot(122)
scatter(testing_data_lda(1:20,1),testing_data_lda(1:20,2),'r')
hold on
scatter(testing_data_lda(21:40,1),testing_data_lda(21:40,2),'g')
hold on
scatter(testing_data_lda(41:60,1),testing_data_lda(41:60,2),'b')
hold on
plot(x,classifier(1)+x*0,'black')
hold on
plot(x,classifier(2)+x*0,'black')

testing_error=0;
for i=1:20
    if testing_data_lda(i,2) < classifier(1)
        testing_error = testing_error + 1;
    end
    if testing_data_lda(i+20,2) > classifier(1) || testing_data_lda(i+20,2) < classifier(2)
        testing_error = testing_error + 1;
    end
    if testing_data_lda(i+40,2) > classifier(2)
        testing_error = testing_error + 1;
    end
end
title('Testing via PCA & LDA')
xlabel('LD1')
ylabel('LD2')
legend('setosa','versicolor','virginica')
sprintf('The training error is %d, and the testing error is %d.',training_error,testing_error)
end
%%

function [v,d] = LDA(data,label)
x1=[];
x2=[];
x3=[];
count1=0;
count2=0;
count3=0;
for i=1:length(label)
    if label(i) == 1
        count1 = count1+1;
        x1(:,count1) = data(:,i);
    elseif label(i) == 2
        count2 = count2+1;
        x2(:,count2) = data(:,i);
    elseif label(i) ==3
        count3 = count3+1;
        x3(:,count3) =data(:,i);
    end
end
m1 = mean(x1(:,:),2);
m2 = mean(x2(:,:),2);
m3 = mean(x3(:,:),2);
moverall = mean(data,2);
S_w = (x1(:,:)-m1)*(x1(:,:)-m1)'+(x2(:,:)-m2)*(x2(:,:)-m2)'+(x3(:,:)-m3)*(x3(:,:)-m3)';
S_b = count1*(m1-moverall)*(m1-moverall)'+count2*(m2-moverall)*(m2-moverall)'+count3*(m3-moverall)*(m3-moverall)';
    
[v,d] = eig(inv(S_w)*S_b);

end