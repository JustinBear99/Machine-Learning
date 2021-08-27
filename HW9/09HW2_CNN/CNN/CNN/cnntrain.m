function net = cnntrain(net, x, y, opts)
    m = size(x, 3);
    numbatches = m / opts.batchsize;
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    net.rL = [];
    net.train_rL= [];
%     net.test_rL=[];
    for i = 1 : opts.numepochs
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        tic;
        kk = randperm(m); %a random permutation of the integers from 1 to m
        for l = 1 : numbatches
%             test=kk;
%             test(:,(l - 1) * opts.batchsize + 1 : l * opts.batchsize)=[];
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:,    kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
%             batch_x2 = x(:,:,test);
%             batch_y2 = y(:,test);
            net = cnnff(net, batch_x);
            net = cnnbp(net, batch_y);
            net = cnnapplygrads(net, opts);
            if isempty(net.rL)
                net.rL(1) = net.L;
            else
                 net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
            end
           
            
            net =cnnff(net,batch_x);
            net.e = net.o - batch_y;
            net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2); 
            if isempty(net.train_rL)
                net.train_rL(1) = net.L;
            else
                 net.train_rL(end + 1) = net.L;
            end
            
%             net = cnnff(net,test_x);
%             net.e = net.o - test_y;
%             net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
%             if isempty(net.test_rL)
%                 net.test_rL(1) = net.L;
%             else
%                  net.test_rL(end + 1) = net.L;
%             end
%            
%             net = cnnff(net, batch_x2);
%             net.e = net.o - batch_y2;
%             net.L = 1/2* sum(net.e(:) .^ 2) / size(net.e, 2);
%              if isempty(net.rrL)
%                 net.rrL(1) = net.L;
%             end
%             net.rrL(end + 1) = 0.99 * net.rrL(end) + 0.01 * net.L;           
        end
%         net.rrL(i,:)=net.rL;
        
        toc;
% % % % % % % % % % % % % % % % % % % % % % % % %         
%         if min(net.rL)<opts.error*0.5
%             break
%         end
    end
    
end
