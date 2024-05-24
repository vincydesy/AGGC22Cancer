% Supponiamo che la tua cell array si chiami 'data' e sia caricata nel workspace
% Caricamento della cell array (modifica il percorso al tuo file .mat se necessario)
load('immagini_info_all.mat'); % Modifica con il percorso del tuo file .mat

% Estrai le caratteristiche e le etichette dalla cell array
num_samples = size(combinedData, 1);
feature_length = size(combinedData{1, 2}, 2); % Assumendo che tutte le caratteristiche abbiano la stessa lunghezza

X = zeros(num_samples, feature_length);
Labels = cell(num_samples, 1);
ImageNames = cell(num_samples, 1);


for i = 1:num_samples
    ImageNames{i} = combinedData{i, 1}; % Assumendo che i nomi delle immagini siano nella prima colonna
    X(i, :) = combinedData{i, 2}; % Assumendo che le caratteristiche siano nella seconda colonna
    Labels{i} = combinedData{i, 3}; % Assumendo che le etichette siano nella terza colonna
end


% Parametri per la funzione IterativeEntropyClustering
MaxIter = 10;
max_cl_size_th = 100;
stop_dth = 0;
MaxLabelTh = 3; % Soglia massima del numero di etichette diverse in un cluster

% Chiamata alla funzione IterativeEntropyClustering
[clusters, dvals] = IterativeEntropyClustering(X, MaxIter, max_cl_size_th, stop_dth);


% Inizializzazione degli array di output
SelectedItems = [];
SelectedLabels = [];
outputFolder = 'clusters'; % Nome della cartella dove salvare i file

% Creazione della cartella se non esiste
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Analisi dei cluster per determinare se soddisfano la soglia delle etichette
for i = 1:length(clusters)
    clusterIndices = clusters{i};  % Indici degli elementi nel cluster corrente
    clusterLabels = Labels(clusterIndices);  % Etichette degli elementi nel cluster corrente
    
    if length(unique(clusterLabels)) <= MaxLabelTh
        % Se il numero di etichette uniche è inferiore alla soglia, aggiungi gli indici e le etichette agli array di output
        SelectedItems = [SelectedItems; clusterIndices];
        SelectedLabels = [SelectedLabels; clusterLabels];
        
        % Salva il cluster in un file .mat
        clusterFeatures = X(clusterIndices, :);
        clusterImageNames = ImageNames(clusterIndices); % Nomi delle immagini nel cluster
        clusterFilename = fullfile(outputFolder, sprintf('cluster_%d.mat', i));
        save(clusterFilename, 'clusterIndices', 'clusterFeatures', 'clusterLabels', 'clusterImageNames');
    end
end

% Report sul numero di elementi selezionati e sulla distribuzione delle etichette
%fprintf('Numero di elementi selezionati: %d\n', length(SelectedItems));
%uniqueLabels = unique(SelectedLabels);
%counts = histc(SelectedLabels, uniqueLabels);
%fprintf('Distribuzione delle etichette nei cluster selezionati:\n');
%for i = 1:length(uniqueLabels)
%    fprintf('Etichetta %d: %d occorrenze\n', uniqueLabels(i), counts(i));
%end
