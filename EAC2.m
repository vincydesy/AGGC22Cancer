% Definisci la cartella delle immagini
cartella_immagini = 'D:\AGGC22_Mio\prova\train';
files = dir(sprintf('%s\\*.tiff', cartella_immagini));
num_immagini = numel(files);

% Caricamento delle immagini nella coda
coda_immagini = cell(1, numel(files));
for i = 1:numel(files)
    filename = files(i).name;
    immagine = imread(fullfile(cartella_immagini, filename));
    coda_immagini{i} = immagine;
end

% Inizializzazione dei cluster
num_clusters = 5;
clusters = cell(num_clusters, 1); % classi delle immagini

% Parametri per l'entropia
soglia_entropia = 7; % Modifica la soglia a seconda delle tue esigenze
max_images_per_cluster = ceil(numel(files) / num_clusters); % Limita il numero di immagini per cluster

% Iterazione sulla coda di immagini
for i = 1:numel(files)
    immagine_corrente = coda_immagini{i};
    entropia_attuale = entropy(immagine_corrente);

    % Variabili per tracciare il miglior cluster
    min_entropia_diff = Inf;
    best_cluster_idx = 0;

    for c = 1:num_clusters
        if ~isempty(clusters{c})
            % Calcola l'entropia media del cluster corrente
            cluster_entropia = 0;
            num_images = numel(clusters{c});
            for k = 1:num_images
                cluster_entropia = cluster_entropia + entropy(clusters{c}{k});
            end
            cluster_entropia = cluster_entropia / num_images;

            % Calcola la differenza di entropia tra l'immagine corrente e il cluster
            entropia_diff = abs(cluster_entropia - entropia_attuale);

            % Aggiorna il miglior cluster se la differenza è minore
            if entropia_diff < min_entropia_diff && num_images < max_images_per_cluster
                min_entropia_diff = entropia_diff;
                best_cluster_idx = c;
            end
        else
            % Se il cluster è vuoto, assegnalo direttamente
            best_cluster_idx = c;
            break;
        end
    end

    % Aggiungi l'immagine corrente al miglior cluster trovato
    if isempty(clusters{best_cluster_idx})
        clusters{best_cluster_idx} = {immagine_corrente};
    else
        clusters{best_cluster_idx}{end + 1} = immagine_corrente;
    end
end

% Creazione delle cartelle
output_folder = 'output_clusters';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

for c = 1:num_clusters
    cluster_folder = fullfile(output_folder, ['Cluster_' num2str(c)]);
    if ~exist(cluster_folder, 'dir')
        mkdir(cluster_folder);
    end
    
    % Salvataggio delle immagini nel cluster
    num_images = numel(clusters{c});
    for j = 1:num_images
        % Costruisci il nome del file di output
        [~, name, ext] = fileparts(files(j).name);
        output_filename = fullfile(cluster_folder, [name,'.tiff']);
        
        % Salva l'immagine
        imwrite(clusters{c}{j}, output_filename);
    end
end

% Stampa il numero di immagini in ogni cluster per verifica
for c = 1:num_clusters
    fprintf('Cluster %d: %d immagini\n', c, numel(clusters{c}));
end