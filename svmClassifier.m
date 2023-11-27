dataset= imageDatastore("Dataset", 'IncludeSubfolders',true, 'LabelSource','foldernames');
%split data menjadi data train dan data tes
[TrainingDataset, ValidationDataset, TestingDataset] = splitEachLabel(dataset, 0.7, 0.15, 0.15);

input_layer_size = [256, 256];
temp = readimage(TrainingDataset, 1);
scaledTemp = imresize(temp, input_layer_size);
[features, visualization] = extractHOGFeatures(scaledTemp);

numofImages = numel(TrainingDataset.Files);
train_features = zeros(numofImages, size(features, 2), 'single');

for i = 1:numofImages
    imageTrain = readimage(TrainingDataset, i);
    imageTrain = imresize(imageTrain, input_layer_size);
    train_features(i, :) = extractHOGFeatures(imageTrain);
end

trainLabels = TrainingDataset.Labels;
SVM_classifier = fitcecoc(train_features, trainLabels);

save SVM_classifier



