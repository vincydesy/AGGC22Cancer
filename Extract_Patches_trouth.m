function Extract_Patches_trouth(src_dire, dst_dire, trouth_path, imaname, npx, lev)

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
        %newFilename = sprintf('%s\\%s_%d_%d.tiff', dst_dire, imaname(1:end-5), startY, startX);

        if(std(double(patch(:)))>10)

            max_white = 0;
            d=dir(sprintf('%s\\*.tif', trouth_path));
            for patch_dir=1:size(d,1)
                filename_mask = sprintf('%s\\%s', trouth_path, d(patch_dir).name);
                mask_name = d(patch_dir).name;
                patch_trouth = imread(filename_mask, 'PixelRegion', {[startY, startY + patchHeight - 1], [startX, startX + patchWidth - 1]});
                conteggio_bianchi = sum(patch_trouth(:));
                if conteggio_bianchi > 0 && conteggio_bianchi > max_white
                    max_white = conteggio_bianchi;
                    componenti = split(mask_name, '_');
                    class = componenti{1};
                end

            end
            % Scrivi la patch in un nuovo file tif se è stata trovata una
            % zona tumorale
            if max_white > 0
            newFilename = sprintf('%s\\%s_%d_%d_%s.tiff', dst_dire, imaname(1:end-5), startY, startX, class);
            imwrite(patch, newFilename);
            end

             %figure(1);
             %imshow(patch_trouth);
             %drawnow;

        end
    end
end