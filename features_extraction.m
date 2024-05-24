% Carica la rete ResNet-50 pre-addestrata
net = resnet50;

% Definisci la cartella delle immagini
cartella_immagini = 'D:\Istologico\script elaborazione patch matlab\clusters\cluster5';

% Definisci la cartella di destinazione per le caratteristiche estratte
cartella_output = 'D:\Istologico\script elaborazione patch matlab\clusters\cluster5';


% Estrai caratteristiche per ogni immagine nella cartella
files = dir(sprintf('%s\\*.tiff', cartella_immagini));
% Inizializza una cella per archiviare le informazioni di ogni immagine
immagini_info = cell(length(files), 3); % 3 colonne: nome immagine, caratteristiche, etichetta
for i = 1:size(files)
    filename = files(i).name;
    [~, ~, ext] = fileparts(filename);
    fprintf("%d\n", i)

        img = imread(fullfile(cartella_immagini, filename));
        img = imresize(img, [224, 224]); % Ridimensiona l'immagine a 224x224 (dimensioni richieste dalla rete ResNet-50)
        
        % Estrai le caratteristiche utilizzando la rete ResNet-50
        features = activations(net, img, 'avg_pool');
        features = squeeze(features);
        features = features(1:1024,:)';
        
        % Estrai l'etichetta dall'immagine (considerando il formato "nome_numeropatch_label")
        label = strsplit(filename, '_');
        label = label{end};
        label = strrep(label, ext, ''); % Rimuovi l'estensione del file
        [~, filename_without_ext, ~] = fileparts(filename);

        % Aggiungi le informazioni dell'immagine alla cella
        immagini_info{i, 1} = filename_without_ext;
        immagini_info{i, 2} = features;
        immagini_info{i, 3} = label;

    
end


% Salva le informazioni delle immagini in un file MAT
save('immagini_info_cluster_5.mat');
