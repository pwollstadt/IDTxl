javaaddpath('/data/home1/pwollsta/jidt_1_2_1/infodynamics.jar');

% Generate some random binary data.
sourceArray=randn(100,1); 
destArray = [0; sourceArray(1:99)];
sourceArray2=rand(100,1);
% Create a TE calculator and run it:
% TransferEntropyCalculatorDiscrete(int base, int destHistoryEmbedLength)
cmiCalc=javaObject('infodynamics.measures.continuous.kraskov.ConditionalMutualInfoCalculatorMultiVariateKraskov2');
cmiCalc.initialise(1,1,1);
cmiCalc.setObservations(sourceArray, destArray, sourceArray2);
result = cmiCalc.computeAverageLocalOfObservations();
fprintf('CMI result: %0.4f\n', result);
