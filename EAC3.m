% Definisci la cartella delle immagini
cartella_immagini = 'D:\AGGC22_Mio\train_set_patches';
files = dir(sprintf('%s\\*.tiff', cartella_immagini));
num_immagini = numel(files);

% Inizializzazione dei cluster
num_clusters = 5;
clusters = cell(num_clusters, 1); % classi delle immagini
entropie_medie = zeros(num_clusters, 1); % entropie medie dei cluster
num_immagini_cluster = zeros(num_clusters, 1); % numero di immagini per cluster

% Parametri per l'entropia
soglia_entropia = 7; % Modifica la soglia a seconda delle tue esigenze
max_images_per_cluster = ceil(numel(files) / num_clusters); % Limita il numero di immagini per cluster

% Iterazione sulla coda di immagini
for i = 1:numel(files)
    filename = files(i).name;
    immagine = imread(fullfile(cartella_immagini, filename));
    immagine_corrente = immagine;
    clear immagine;
    entropia_attuale = entropy(immagine_corrente);
    sprintf("%d\n",i)

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

    % Aggiungi l'immagine corrente al miglior cluster trovato
    if num_immagini_cluster(best_cluster_idx) == 0
        clusters{best_cluster_idx} = {immagine_corrente};
        entropie_medie(best_cluster_idx) = entropia_attuale;
        num_immagini_cluster(best_cluster_idx) = 1;
    else
        clusters{best_cluster_idx}{end + 1} = immagine_corrente;
        num_immagini_cluster(best_cluster_idx) = num_immagini_cluster(best_cluster_idx) + 1;
        % Aggiorna l'entropia media del cluster
        entropie_medie(best_cluster_idx) = ((entropie_medie(best_cluster_idx) * (num_immagini_cluster(best_cluster_idx) - 1)) + entropia_attuale) / num_immagini_cluster(best_cluster_idx);
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