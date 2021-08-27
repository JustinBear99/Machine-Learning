Y=[-0.4 0.58 0.089 0.83 1.6 -0.014;
    -0.31 0.27 -0.04 1.1 1.6 0.48;
    0.38 0.055 -0.035 -0.44 -0.41 0.32;
    -0.15 0.53 0.011 0.047 -0.45 0.32;
    -0.35 0.47 0.034 0.28 0.35 3.1;
    0.17 0.69 0.1 -0.39 -0.48 0.11;
    -0.011 0.55 -0.18 0.34 -0.079 0.14;
    -0.27 0.61 0.12 -0.3 -0.22 2.2;
    -0.065 0.49 0.0012 1.1 1.2 -0.46;
    -0.12 0.054 -0.063 0.18 -0.11 -0.49];

data = [Y(:,1:3)' Y(:,4:6)'];
label = [1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2];
[v,d] = LDA(data,label);

x1 = data(1,:)';
x2 = data(2,:)';
x3 = data(3,:)';
y = label;
S = ones(20,1)*20;
c = [repmat([1 0 0],10,1) ;repmat([0 1 0],10,1)];
figure
scatter3(x1,x2,x3,S,c)
hold on
q=quiver3(0,0,0,v(1),v(2),v(3));
q.LineWidth = 2;
%%
new_d = [d(:,1) d(:,2)];

data_lda = data'*new_d;
classfier = (median(data_lda(1:10,2)) + median(data_lda(11:20,2)))/2;
y=linspace(-1,1);
scatter(data_lda(1:10,2),zeros(10,1),'r')
hold on
scatter(data_lda(11:20,2),zeros(10,1),'g')
hold on

plot(classfier+y*0,y,'black')
title('1d projection')
xlabel('LC1')
training_error=0;
for i=1:10
    if data_lda(i,2) < classfier
        training_error = training_error +1;
    end
    if data_lda(i+10,2) > classfier
        training_error = training_error +1;
    end
end
sprintf('The training error is %d.',training_error)
function [v,d] = LDA(data,label)
x1=[];
x2=[];
count1=0;
count2=0;
for i=1:length(label)
    if label(i) == 1
        count1 = count1+1;
        x1(:,count1) = data(:,i);
    elseif label(i) == 2
        count2 = count2+1;
        x2(:,count2) = data(:,i);
    end
end
m1 = mean(x1(:,:),2);
m2 = mean(x2(:,:),2);
moverall = mean(data,2);
S_w = (x1(:,:)-m1)*(x1(:,:)-m1)'+(x2(:,:)-m2)*(x2(:,:)-m2)';
S_b = count1*(m1-moverall)*(m1-moverall)'+count2*(m2-moverall)*(m2-moverall)';
[v,d] = eig(inv(S_w)*S_b);

end