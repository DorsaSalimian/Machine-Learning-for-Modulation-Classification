clc; clear; close all;

M_list = {'BPSK','QPSK','QAM16'};
numSignalsPerClass = 500;
SNR_dB = 10;
N = 1024;   

Features = [];
Labels = [];

for m = 1:length(M_list)
    for k = 1:numSignalsPerClass

        bits = randi([0 1], N, 1);

        switch M_list{m}
            case 'BPSK'
                symbols = 2*bits - 1;

            case 'QPSK'
                bits2 = reshape(bits(1:end-mod(length(bits),2)),2,[]);
                symbols = (2*bits2(1,:)-1) + 1j*(2*bits2(2,:)-1);
                symbols = symbols.'/sqrt(2);

            case 'QAM16'
                bits4 = reshape(bits(1:end-mod(length(bits),4)),4,[]);
                symbols = qammod(bi2de(bits4.'),16,'UnitAveragePower',true);
        end

        rx = awgn(symbols,SNR_dB,'measured');

        f = extract_features(rx);

        Features = [Features; f];
        Labels   = [Labels; m];
    end
end

X = Features;
Y = categorical(Labels);

cv = cvpartition(Y,'HoldOut',0.3);
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv));
Xtest  = X(test(cv),:);
Ytest  = Y(test(cv));

SVMmodel = fitcecoc(Xtrain,Ytrain);
Ypred = predict(SVMmodel,Xtest);

accuracy = mean(Ypred == Ytest)*100;
fprintf('SVM Accuracy = %.2f %%\n',accuracy);

figure;
confusionchart(Ytest,Ypred);
title('SVM Modulation Classification');


function f = extract_features(x)
    x = x(:);
    amp = abs(x);

    f1 = mean(amp);
    f2 = var(amp);
    f3 = kurtosis(amp);
    f4 = skewness(amp);
    f5 = mean(amp.^4);

    f = [f1 f2 f3 f4 f5];
end
