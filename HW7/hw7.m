data = load('07HW2_digit.mat');
%colormap gray;
%imagesc(~reshape(train0(1,:), 28, 28)');
training_instance_matrix = double([data.train0; data.train1; data.train2; data.train3; data.train4; data.train5; data.train6; data.train7; data.train8; data.train9]);
training_label_vector = [zeros(500,1); ones(500,1); ones(500,1)*2; ones(500,1)*3; ones(500,1)*4; ones(500,1)*5; ones(500,1)*6; ones(500,1)*7; ones(500,1)*8; ones(500,1)*9];
testing_instance_matrix = double([data.test0; data.test1; data.test2; data.test3; data.test4; data.test5; data.test6; data.test7; data.test8; data.test9]);
testing_label_vector = [zeros(100,1); ones(100,1); ones(100,1)*2; ones(100,1)*3; ones(100,1)*4; ones(100,1)*5; ones(100,1)*6; ones(100,1)*7; ones(100,1)*8; ones(100,1)*9];
%%
% a)
training_instance_matrix_a = double([data.train0; data.train1]);
training_label_vector_a = [zeros(500,1); ones(500,1)];
testing_instance_matrix_a = double([data.test0; data.test1]);
testing_label_vector_a = [zeros(100,1); ones(100,1)];
gamma = [2e-14; 2e-12; 2e-10; 2e-8; 2e-6;];
c = [2e-5; 2e-3; 2e-1; 2; 2e3;];
for i=1:size(gamma,1)
    for j=1:size(c,1)
        options(i,j) = strcat('-g',{' '}, string(gamma(i)),{' '}, '-c',{' '}, string(c(j)));
    end
end
output = zeros(5,5); 
for i=1:5
    for j=1:5
        model = svmtrain(training_label_vector_a,training_instance_matrix_a,options);
        [predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_a, testing_instance_matrix_a, model);
        output(i,j) = accuracy(1);
    end
end
%%
% b)
%linear
model = svmtrain(training_label_vector_a,training_instance_matrix_a,'-t 0 -q');
linearoutput = accuracy(1);
%poly
degree = [3;5;7;9;11];
for i=1:size(gamma,1)
    for j=1:size(degree,1)
        options2(i,j) = strcat('-g',{' '}, string(gamma(i)),{' '}, '-d',{' '}, string(degree(j)));
    end
end
output2 = zeros(5,5);
for i=1:5
    for j=1:5
        model = svmtrain(training_label_vector_a,training_instance_matrix_a,options2);
        [predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_a, testing_instance_matrix_a, model);
        output2(i,j) = accuracy(1);
    end
end

%%
% c)
training_label_vector_0 = [zeros(500,1); ones(4500,1)];
testing_label_vector_0 = [zeros(100,1); ones(900,1)];
model = svmtrain(training_label_vector_0,training_instance_matrix,'-t 0 -q');
[predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_0, testing_instance_matrix, model);
output3(1) = accuracy(1);

training_label_vector_1 = [ones(500,1); zeros(500,1); ones(4000,1)];
testing_label_vector_1 = [ones(100,1); zeros(100,1); ones(800,1)];
model = svmtrain(training_label_vector_1,training_instance_matrix,'-t 0 -q');
[predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_1, testing_instance_matrix, model);
output3(2) = accuracy(1);

training_label_vector_2 = [ones(1000,1); zeros(500,1); ones(3500,1)];
testing_label_vector_2 = [ones(200,1); zeros(100,1); ones(700,1)];
model = svmtrain(training_label_vector_2,training_instance_matrix,'-t 0 -q');
[predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_2, testing_instance_matrix, model);
output3(3) = accuracy(1);

training_label_vector_3 = [ones(1500,1); zeros(500,1); ones(3000,1)];
testing_label_vector_3 = [ones(300,1); zeros(100,1); ones(600,1)];
model = svmtrain(training_label_vector_3,training_instance_matrix,'-t 0 -q');
[predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_3, testing_instance_matrix, model);
output3(4) = accuracy(1);

training_label_vector_4 = [ones(2000,1); zeros(500,1); ones(2500,1)];
testing_label_vector_4 = [ones(400,1); zeros(100,1); ones(500,1)];
model = svmtrain(training_label_vector_4,training_instance_matrix,'-t 0 -q');
[predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_4, testing_instance_matrix, model);
output3(5) = accuracy(1);

training_label_vector_5 = [ones(2500,1); zeros(500,1); ones(2000,1)];
testing_label_vector_5 = [ones(500,1); zeros(100,1); ones(400,1)];
model = svmtrain(training_label_vector_5,training_instance_matrix,'-t 0 -q');
[predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_5, testing_instance_matrix, model);
output3(6) = accuracy(1);

training_label_vector_6 = [ones(3000,1); zeros(500,1); ones(1500,1)];
testing_label_vector_6 = [ones(600,1); zeros(100,1); ones(300,1)];
model = svmtrain(training_label_vector_6,training_instance_matrix,'-t 0 -q');
[predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_6, testing_instance_matrix, model);
output3(7) = accuracy(1);

training_label_vector_7 = [ones(3500,1); zeros(500,1); ones(1000,1)];
testing_label_vector_7 = [ones(700,1); zeros(100,1); ones(200,1)];
model = svmtrain(training_label_vector_7,training_instance_matrix,'-t 0 -q');
[predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_7, testing_instance_matrix, model);
output3(8) = accuracy(1);

training_label_vector_8 = [ones(4000,1); zeros(500,1); ones(500,1)];
testing_label_vector_8 = [ones(800,1); zeros(100,1); ones(100,1)];
model = svmtrain(training_label_vector_8,training_instance_matrix,'-t 0 -q');
[predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_8, testing_instance_matrix, model);
output3(9) = accuracy(1);

training_label_vector_9 = [ones(4500,1); zeros(500,1)];
testing_label_vector_9 = [ones(900,1); zeros(100,1)];
model = svmtrain(training_label_vector_9,training_instance_matrix,'-t 0 -q');
[predicted_label, accuracy, decision_values] = svmpredict(testing_label_vector_9, testing_instance_matrix, model);
output3(10) = accuracy(1);

