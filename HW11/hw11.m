load 11HW1_KmeanData.mat
X = X';
Y = Y';
X4 = kmeans(X,4);  X5 = kmeans(X,5);  X6 = kmeans(X,6);
subplot(231)
gscatter(X(:,1),X(:,2),X4);
title('X into 4 clusters')
subplot(232)
gscatter(X(:,1),X(:,2),X5);
title('X into 5 clusters')
subplot(233)
gscatter(X(:,1),X(:,2),X6);
title('X into 6 clusters')
Y4 = kmeans(Y,4);  Y5 = kmeans(Y,5);  Y6 = kmeans(Y,6);
subplot(234)
gscatter(Y(:,1),Y(:,2),Y4);
title('Y into 4 clusters')
subplot(235)
gscatter(Y(:,1),Y(:,2),Y5);
title('Y into 5 clusters')
subplot(236)
gscatter(Y(:,1),Y(:,2),Y6);
title('Y into 6 clusters')