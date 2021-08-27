load('04HW2_noisy.mat');
[U,S,V] = svd(X);
newS = zeros(size(S));
for i=1:6
        newS(i,i) = S(i,i);
end
recordS = zeros(560,1);
for j=1:560
    recordS(j) = S(j,j);
end
newX = U*newS*V;
colormap gray
subplot(2,5,1)
imagesc(reshape(X(:, 10), 20, 28)')
subplot(2,5,2)
imagesc(reshape(X(:, 121), 20, 28)')
subplot(2,5,3)
imagesc(reshape(X(:, 225), 20, 28)')
subplot(2,5,4)
imagesc(reshape(X(:, 318), 20, 28)')
subplot(2,5,5)
imagesc(reshape(X(:, 426), 20, 28)')
subplot(2,5,6)
imagesc(reshape(newX(:, 10), 20, 28)')
subplot(2,5,7)
imagesc(reshape(newX(:, 121), 20, 28)')
subplot(2,5,8)
imagesc(reshape(newX(:, 225), 20, 28)')
subplot(2,5,9)
imagesc(reshape(newX(:, 318)*-1, 20, 28)')
subplot(2,5,10)
imagesc(reshape(newX(:, 426)*-1, 20, 28)')
%%
sum_S = sum(recordS);
recordS(:,2) = recordS(:,1)./sum_S;
recordS(1,3) = recordS(1,2);
for i=2:560
    recordS(i,3) = recordS(i,2)+recordS(i-1,3);
end
%%
plot(1:15,recordS(1:15,2),1:15,recordS(1:15,3))
xlabel('intrinsic dimensionality')
ylabel('percentage')
title('Importance of the first 50 eigenvalues')
legend('each','cumulative')