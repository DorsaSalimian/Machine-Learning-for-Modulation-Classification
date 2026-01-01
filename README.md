# Machine-Learning-for-Modulation-Classification
Automatic Modulation Recognition/Classification in communication systems

🎯 Overall Goal & Final Objective

Problem: Automatic Modulation Recognition/Classification in communication systems

Input: Noisy received signal (unknown modulation type)

Output: Predicted modulation type (BPSK, QPSK, or 16-QAM)

Final Achievement: SVM classifier with 98.22% accuracy on test data

Phase 1: Data Generation (Synthetic Dataset Creation)

M_list = {'BPSK','QPSK','QAM16'};        % 3 modulation types
numSignalsPerClass = 500;                % 500 signals per class
SNR_dB = 10;                             % Signal-to-Noise Ratio = 10dB
N = 1024;                                % Signal length = 1024 symbols


For each modulation type × each signal:

    Generate random bits: randi([0 1], N, 1)
    Modulate bits → complex symbols
    Add AWGN noise: awgn(symbols, SNR_dB)
    Extract 5 statistical features
    Store features + true label

Modulation Implementation Details:

Modulation    	Bits per Symbol 	           Implementation
BPSK 	               1                  	symbols = 2*bits - 1 (→ ±1)
QPSK 	               2 	                  (±1±j1)/√2 (4 points on circle)
16-QAM 	             4 	                  qammod(bi2de(bits4),16) (16 points in square)

Total Dataset: 3×500=1500 signals → 4500 features 

Phase 2: Feature Extraction (The KEY Innovation)

function f = extract_features(rx)
    amp = abs(rx);  % Get amplitude envelope
    f = [mean(amp), var(amp), kurtosis(amp), skewness(amp), mean(amp.^4)];
end

Phase 3: Train/Test Split

cv = cvpartition(Y,'HoldOut',0.3);  % 70% train (1050), 30% test (450)

Phase 4: SVM Multi-Class Classification

SVMmodel = fitcecoc(Xtrain,Ytrain);  % ECOC (Error-Correcting Output Codes)
Ypred = predict(SVMmodel,Xtest);

How ECOC works for 3 classes:

Class 1: [+ + -]

Class 2: [+ - +]

Class 3: [- + +]

→ SVM solves 3 binary problems → Voting gives final class

Phase 5: Evaluation & Visualization

accuracy = 98.22%
