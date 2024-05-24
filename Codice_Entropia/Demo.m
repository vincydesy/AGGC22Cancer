% % Spiegazioni dello Script
% % Generazione di Dati Casuali: Lo script inizia con la creazione di una matrice X di 1000 righe e 1024 colonne con valori casuali. 
% %Inoltre, genera un array Labels con 1000 elementi, ciascuno con un valore casuale tra 1 e 9, che rappresentano le etichette dei dati in X.
% % 
% % Configurazione e Esecuzione del Clustering: Viene definita una funzione di clustering fittizia IterativeEntropyClustering, 
% % richiamata con parametri specifici per iterare fino a 10 volte, con una dimensione massima del cluster di 100 e una 
% %soglia di arresto dello spostamento a 0.
% % 
% % Selezione dei Cluster: Per ciascun cluster risultante, verifica se il numero di etichette uniche è inferiore o 
% % uguale a una soglia (3 nel caso di questo script). Se è così, indice e etichette del cluster vengono salvati in due array di output, 
% % SelectedItems e SelectedLabels.
% % 
% % Reporting: Infine, lo script stampa il numero totale di elementi selezionati e una distribuzione delle etichette tra questi.

% Inizializzazione di un array X con 1000 righe e 1024 colonne di dati casuali
X = rand(1000, 1024);

% Generazione di un array Labels con 1000 elementi con valori nell'intervallo [1,9]
Labels = randi([1, 9], 1000, 1);

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

% Analisi dei cluster per determinare se soddisfano la soglia delle etichette
for i = 1:length(clusters)
    clusterIndices = clusters{i};  % Indici degli elementi nel cluster corrente
    clusterLabels = Labels(clusterIndices);  % Etichette degli elementi nel cluster corrente
    
    if length(unique(clusterLabels)) <= MaxLabelTh
        % Se il numero di etichette uniche è inferiore alla soglia, aggiungi gli indici e le etichette agli array di output
        SelectedItems = [SelectedItems; clusterIndices];
        SelectedLabels = [SelectedLabels; clusterLabels];
    end
end

% Report sul numero di elementi selezionati e sulla distribuzione delle etichette
fprintf('Numero di elementi selezionati: %d\n', length(SelectedItems));
uniqueLabels = unique(SelectedLabels);
counts = histc(SelectedLabels, uniqueLabels);
fprintf('Distribuzione delle etichette nei cluster selezionati:\n');
for i = 1:length(uniqueLabels)
    fprintf('Etichetta %d: %d occorrenze\n', uniqueLabels(i), counts(i));
end
