function trainResNetOnCriterion(imageDir)
    % Funzione per estrarre l'etichetta dal nome dell'immagine
    function label = extractLabelFromFilename(filename)
        parts = strsplit(filename, '_');
        label = parts{end-1};  % Assumendo che l'estensione .tiff sia l'ultima parte
    end

    parpool(4);

    % Creare un datastore per le immagini
    imds = imageDatastore(imageDir, 'FileExtensions', '.tiff');

    % Estrarre le etichette dai nomi dei file
    labels = cellfun(@extractLabelFromFilename, imds.Files, 'UniformOutput', false);
    imds.Labels = categorical(labels);

    % Dividere i dati in set di allenamento e di valutazione
    [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

    % Dimensioni desiderate delle immagini
    targetSize = [224 224];

    % Ridimensionare le immagini di addestramento
    imdsTrainResized = augmentedImageDatastore(targetSize, imdsTrain);

    % Ridimensionare le immagini di validazione
    imdsValidationResized = augmentedImageDatastore(targetSize, imdsValidation);

    % Caricare una ResNet pre-addestrata
    net = resnet50;
    lgraph = layerGraph(net);

    % Modificare i layer finali
    numClasses = numel(categories(imdsTrain.Labels));
    newFCLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
    newClassLayer = classificationLayer('Name', 'new_classoutput');

    lgraph = replaceLayer(lgraph, 'fc1000', newFCLayer);
    lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassLayer);

    % Impostare le opzioni di allenamento
    options = trainingOptions('sgdm', ...
        'MiniBatchSize', 32, ...
        'MaxEpochs', 10, ...
        'InitialLearnRate', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', imdsValidationResized, ...
        'ValidationFrequency', 30, ...
        'Verbose', false, ...
        'Plots', 'training-progress');

    % Allenare la rete
    trainedNet = trainNetwork(imdsTrainResized, lgraph, options);

    % Salvare il modello addestrato
    [~, criterionName, ~] = fileparts(imageDir);
    modelFilename = sprintf('trainedResNet_%s.mat', criterionName);
    save(modelFilename, 'trainedNet');

    delete(gcp('nocreate'));
end
