% Directory contenente i file cluster_x.mat
clusterDir = 'D:\Istologico\script elaborazione patch matlab\clusters';
% Directory contenente le immagini
imageDir = 'D:\AGGC22_Mio\train_set_patches';
% Directory di destinazione
outputDir = 'criterio_2';

% Creare la directory di output se non esiste
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Iterare attraverso tutti i file cluster_x.mat
for i = 1:6000
    % Caricare il file cluster_x.mat
    clusterFile = fullfile(clusterDir, sprintf('cluster_%d.mat', i));
    if exist(clusterFile, 'file')
        data = load(clusterFile);
        
        % Estrarre le etichette e i nomi delle immagini
        clusterLabels = data.clusterLabels;
        clusterImageNames = data.clusterImageNames;

        % Convertire le etichette in categoriali se non lo sono già
        if ~iscategorical(clusterLabels)
            clusterLabels = categorical(clusterLabels);
        end
        
         % Determinare l'etichetta di maggioranza
        uniqueLabels = categories(clusterLabels);
        labelCounts = countcats(clusterLabels);
        [~, majorityIndex] = max(labelCounts);
        majorityLabel = uniqueLabels{majorityIndex};
        
        % Copiare solo le immagini con l'etichetta di maggioranza
        for j = 1:length(clusterLabels)
            if (majorityLabel =10= clusterLabels(j))
                imageName = clusterImageNames{j};
                % Aggiungere l'estensione.tiff
                imageNameWithExtension = [imageName, '.tiff'];
                sourceImage = fullfile(imageDir, imageNameWithExtension);
                destinationImage = fullfile(outputDir, imageNameWithExtension);
                
                % Verificare se l'immagine esiste nella cartella delle immagini
                if exist(sourceImage, 'file')
                    copyfile(sourceImage, destinationImage);
                else
                    fprintf('Immagine %s non trovata.\n', imageNameWithExtension);
                end
            end
        end
    else
        fprintf('File %s non trovato.\n', clusterFile);
    end
end

disp('Operazione completata.');
