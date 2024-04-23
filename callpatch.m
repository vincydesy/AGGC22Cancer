% Scaricare tutte le immagini dal sito. Una alla volta, poiché, a causa della dimensione, zip corrompe i file quando li comprime su drive e dopo averli scaricati non è più in grado di estrarli. Se dal sito si scarica una immagine alla volta non coprime i file. 
% Supponendo che la cartella Dire contenga un insieme di immagini tiff
Dire = 'D:\boh';
src_dire = Dire; %(es. Dire='E:\Immagini')
dst_dire = 'D:\boh\patches';
lev = 40;
pxd = 768;
d=dir(sprintf('%s\\*.tiff', Dire));
for i=1:size(d,1)
   [~, nome_senza_estensione, ~] = fileparts(d(i).name);
   trouth_path = sprintf("%s\\%s", Dire, nome_senza_estensione);
   Extract_Patches_trouth(src_dire, dst_dire, trouth_path, d(i).name, pxd, lev);
end

% lev è il livello da 5x a 40x
% pxd è la dimensione delle patch, ovvero pxd x pxd (es. 512x512);