% Directory contenente i file cluster_x.mat
clusterDir = 'D:\Istologico\script elaborazione patch matlab\clusters';
% Directory contenente le immagini
imageDir = 'D:\AGGC22_Mio\train_set_patches';
% Directory di destinazione
outputDir = 'criterio_1';

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
        
        % Verificare se tutte le etichette sono uguali
        if all(strcmp(clusterLabels, clusterLabels{1}))
            % Copiare le immagini corrispondenti nella cartella di destinazione
            for j = 1:length(clusterImageNames)
                imageName = clusterImageNames{j};
                imageName = [imageName, '.tiff'];
                sourceImage = fullfile(imageDir, imageName);
                destinationImage = fullfile(outputDir, imageName);
                
                % Verificare se l'immagine esiste nella cartella delle immagini
                if exist(sourceImage, 'file')
                    copyfile(sourceImage, destinationImage);
                else
                    fprintf('Immagine %s non trovata.\n', imageName);
                end
            end
        end
    else
        fprintf('File %s non trovato.\n', clusterFile);
    end
end

disp('Operazione completata.');
