% Definisci la cartella delle immagini
cartella_immagini = 'D:\AGGC22_Mio\prova\train';
files = dir(sprintf('%s\\*.tiff', cartella_immagini));
num_immagini = numel(files);

% Inizializzazione della coda di immagini
coda_immagini = cell(1, num_immagini);

% Caricamento delle immagini nella coda
for i = 1:size(files)
    filename = files(i).name;
    immagine = imread(fullfile(cartella_immagini, filename));
    coda_immagini{i} = immagine;
end

% Inizializzazione dei cluster
clusters = cell(5,num_immagini); %classi delle immagini
num_clusters = 1;

% Parametri per l'entropia
soglia_entropia = 7; % Modifica la soglia a seconda delle tue esigenze

% Iterazione sulla coda di immagini
for i = 1:size(files)
    immagine_corrente = coda_immagini{i};

    if i<6
        clusters{i,1} = [clusters{i,1}, immagine_corrente];
    else
    
    % Calcolo dell'entropia
    % Qui devi implementare una funzione per calcolare l'entropia dell'immagine corrente
    entropia_attuale = entropy(immagine_corrente); 

    %max = 1000000;
    entropia_cluster=entropia_attuale;
    for c = 1:size(clusters,1)
        for j = 1:size(clusters,2) 
            if ~isempty(clusters{c,j})
                entropia_cluster = entropia_cluster + entropy(clusters{c,j});
                add_cluster_column = j+1;
            end
        end
            entropia_media = mean(entropia_cluster);
            fprintf("%d,%d\n", c,entropia_media)
            if entropia_media < entropia_attuale
                add_cluster_row = c;
            end

    end

        clusters{add_cluster_row,add_cluster_column} = [clusters{add_cluster_row,add_cluster_column}, immagine_corrente];

    end
    
   

    
    % Se l'entropia supera la soglia, considera la fine di un cluster
    %if entropia > soglia_entropia
    %    j=1
     %   num_clusters = num_clusters + 1;
      %  clusters{num_clusters, j} = immagine_corrente;
      %  j=j+1
   % else
    %    % Aggiungi l'immagine al cluster corrente
     %   clusters{num_clusters,j} = [clusters{num_clusters}, immagine_corrente];
      %  j=j+1
    % end
end


