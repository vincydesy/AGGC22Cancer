function Extract_Patches(src_dire, dst_dire, imaname, npx, lev)
% imaname = 'Subset1_Test_1.tiff';
% src_dire = 'F:';
% dst_dire = 'F:\Patches';

filename = sprintf('%s\\%s', src_dire, imaname);

% Ottieni informazioni sull'immagine
info = imfinfo(filename);
imgWidth = info.Width;
imgHeight = info.Height;

lev = max(5, min(lev,40));

% Dimensioni delle patch
patchSize = 40/lev * [npx, npx];

% Calcola il numero di patch in orizzontale e verticale
numPatchesX = ceil(imgWidth / patchSize(2));
numPatchesY = ceil(imgHeight / patchSize(1));


% Leggi e scrivi ogni patch
for i = 0:numPatchesY - 1
    for j = 0:numPatchesX - 1
        % Coordinate dell'angolo superiore sinistro della patch
        startX = j * patchSize(2) + 1;
        startY = i * patchSize(1) + 1;
        
        % Assicurati di non superare i bordi dell'immagine
        if startX + patchSize(2) - 1 > imgWidth
            patchWidth = imgWidth - startX + 1;
        else
            patchWidth = patchSize(2);
        end
        if startY + patchSize(1) - 1 > imgHeight
            patchHeight = imgHeight - startY + 1;
        else
            patchHeight = patchSize(1);
        end
        
        % Leggi la patch dall'immagine
        patch = imread(filename, 'PixelRegion', {[startY, startY + patchHeight - 1], [startX, startX + patchWidth - 1]});
        
        patch = imresize(patch, lev/40);

        % Nome del nuovo file per la patch
        newFilename = sprintf('%s\\%s_%d_%d.tiff', dst_dire, imaname(1:end-5), startY, startX);

        if(std(double(patch(:)))>10)
            % Scrivi la patch in un nuovo file tif
            imwrite(patch, newFilename);
            
            % figure(1);
            % imshow(patch);
            % drawnow;
        end
    end
end
