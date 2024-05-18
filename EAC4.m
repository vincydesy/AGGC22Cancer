% Definisci la cartella delle immagini
cartella_immagini = 'D:\AGGC22_Mio\train_set_patches';
files = dir(sprintf('%s\\*.tiff', cartella_immagini));
num_immagini = numel(files);

% Inizializzazione dei cluster
num_clusters = 5;
cluster_dir = 'clusters';
if ~exist(cluster_dir, 'dir')
    mkdir(cluster_dir);
end
for c = 1:num_clusters
    cluster_subdir = fullfile(cluster_dir, ['cluster' num2str(c)]);
    if ~exist(cluster_subdir, 'dir')
        mkdir(cluster_subdir);
    end
end

entropie_medie = zeros(num_clusters, 1); % entropie medie dei cluster
num_immagini_cluster = zeros(num_clusters, 1); % numero di immagini per cluster

% Parametri per l'entropia
soglia_entropia = 7; % Modifica la soglia a seconda delle tue esigenze
max_images_per_cluster = ceil(numel(files) / num_clusters); % Limita il numero di immagini per cluster

% Iterazione sulla coda di immagini
for i = 1:numel(files)
    filename = files(i).name;
    full_file_path = fullfile(cartella_immagini, filename);
    immagine = imread(full_file_path);
    entropia_attuale = entropy(immagine);
    clear immagine; % Rilascia la memoria usata dall'immagine
    sprintf("%d\n", i)

    % Variabili per tracciare il miglior cluster
    min_entropia_diff = Inf;
    best_cluster_idx = 0;

    for c = 1:num_clusters
        if num_immagini_cluster(c) > 0
            % Calcola la differenza di entropia tra l'immagine corrente e il cluster
            entropia_diff = abs(entropie_medie(c) - entropia_attuale);

            % Aggiorna il miglior cluster se la differenza è minore
            if entropia_diff < min_entropia_diff && num_immagini_cluster(c) < max_images_per_cluster
                min_entropia_diff = entropia_diff;
                best_cluster_idx = c;
            end
        else
            % Se il cluster è vuoto, assegnalo direttamente
            best_cluster_idx = c;
            break;
        end
    end

    % Aggiorna l'entropia media del cluster e il numero di immagini
    if num_immagini_cluster(best_cluster_idx) == 0
        entropie_medie(best_cluster_idx) = entropia_attuale;
    else
        entropie_medie(best_cluster_idx) = ((entropie_medie(best_cluster_idx) * num_immagini_cluster(best_cluster_idx)) + entropia_attuale) / (num_immagini_cluster(best_cluster_idx) + 1);
    end
    num_immagini_cluster(best_cluster_idx) = num_immagini_cluster(best_cluster_idx) + 1;

    % Copia l'immagine nella cartella del cluster con lo stesso nome
    copyfile(full_file_path, fullfile(cluster_dir, ['cluster' num2str(best_cluster_idx)], filename));
end
