%function that accepts parameters architecture (associative memory + classification or only classification)
function [] = train_network(architecture, fun)

%P is a matrix of test inputs [256, 1000]
%T is a target matrix [256, 1000]
%T1 is a target matrix [10, 1000]
load('P_1000.mat');
load('T.mat');
load('T1.mat');

    %a+c
    if (architecture == 1)
        
        %perform associative memory
        %calculate weights and multiply weights with inputs 
        %T = target matrix
        %P = inputs(training set)
        W = T * pinv(P_1000)
        P2 = W * P_1000
        
        %start training net
        net = perceptron;
        net = configure(net,P2,T1);
        
        net.IW{1,1} = rand(10,256); %generates random matrix
        net.b{1,1} = rand(10,1);
        net.divideParam.trainRatio = 85/100; %training
        net.divideParam.valRatio = 15/100; %validation
        
        %hardlim
        if (fun == 1) 
            
            net.performFcn = 'sse'; %criterion

            [net, tr] = train(net, P2, T1);
            
            trainedHardlimAC = net;
            save trainedHardlimAC;
        
        %linear 
        elseif (fun == 2)
            
            net.layers{1}.transferFcn = 'purelin';
            net.inputWeights{1}.learnFcn = 'learngd';       
            net.biases{1}.learnFcn = 'learngd'; 
            net.trainFcn = 'traingd';
            net.performFcn = 'mse';
            
            [net,tr] = train(net,P2,T1);
            
            trainedLinearAC = net
            
            save trainedLinearAC;
            
        %sigmoidal    
        elseif (fun == 3)
            
            net.layers{1}.transferFcn = 'logsig';
            net.inputWeights{1}.learnFcn = 'learngd';       
            net.biases{1}.learnFcn = 'learngd'; 
            net.trainFcn = 'traingd';
            
            net.performFcn = 'mse';
            
            [net,tr]=train(net,P2,T1);
            
            trainedSigmoidAC = net;
            
            save trainedSigmoidAC;
            
        end      
    %c    
    else
        
        %start training net
        net = perceptron;
        net = configure(net,P_1000,T1);
        
        net.IW{1,1} = rand(10,256); %generates random matrix
        net.b{1,1} = rand(10,1);
        net.divideParam.trainRatio = 85/100; %training
        net.divideParam.valRatio = 15/100; %validation
        
        %hardlim
        if (fun == 1) 
            
            net.performFcn ='sse';
      
            [net,tr]=train(net, P_1000, T1);

            trainedHardlim = net;
            
            save trainedHardlim;
            
        %linear 
        elseif (fun == 2)
        
            net.layers{1}.transferFcn = 'purelin';
            net.inputWeights{1}.learnFcn = 'learngd';       
            net.biases{1}.learnFcn = 'learngd'; 
            net.trainFcn = 'traingd';
            net.performFcn = 'mse';
            
            [net,tr] = train(net, P_1000, T1);
            
            trainedLinear = net
            
            save trainedLinear;
            
        %sigmoidal    
        elseif (fun == 3)
            
            net.layers{1}.transferFcn = 'logsig';
            net.inputWeights{1}.learnFcn = 'learngd';       
            net.biases{1}.learnFcn = 'learngd'; 
            net.trainFcn = 'traingd';
            
            net.performFcn = 'mse';
            
            [net, tr]=train(net, P_1000, T1);
            
            trainedSigmoid = net;
            
            save trainedSigmoid;
            
        end
    end
end

