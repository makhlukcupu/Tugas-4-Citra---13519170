dataset= imageDatastore("Dataset", 'IncludeSubfolders',true, 'LabelSource','foldernames');
%split data menjadi data train dan data tes
[TrainingDataset, ValidationDataset, TestingDataset] = splitEachLabel(dataset, 0.7, 0.15, 0.15);

%pretrained CNN yang digunakan adalah googlenet
net = googlenet;
%analyzeNetwork(net);

%resize data gambar untuk cnn
input_layer_size = net.Layers(1).InputSize(1:2);
resized_training_dataset = augmentedImageDatastore(input_layer_size, TrainingDataset);
resized_validation_dataset = augmentedImageDatastore(input_layer_size, ValidationDataset);
resized_testing_dataset = augmentedImageDatastore(input_layer_size, TestingDataset);

feature_learner = net.Layers(142).Name;
output_classifier = net.Layers(144).Name;
number_of_classes = numel(categories(TrainingDataset.Labels));

new_feature = fullyConnectedLayer(number_of_classes, ...
    'Name', 'Vehicle Feature Learner', ...
    "WeightLearnRateFactor", 10, ...
    'BiasLearnRateFactor', 10);
new_classifier_layer = classificationLayer("Name","Vehicle Classifier");
network_architecture = layerGraph(net);
new_network = replaceLayer(network_architecture, feature_learner, new_feature);
new_network = replaceLayer(new_network, output_classifier, new_classifier_layer);

%analyzeNetwork(new_network);
minibatch_size = 4;
validation_frequency = floor(numel(resized_validation_dataset.Files)/minibatch_size);
training_options = trainingOptions('sgdm',...
    'MiniBatchSize', 4, ...
    'MaxEpochs', 6, ...
    'InitialLearnRate', 3e-4, ...
    'Shuffle', 'every-epoch',...
    'ValidationData',resized_validation_dataset, ...
    'ValidationFrequency', validation_frequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');
net = trainNetwork(resized_training_dataset, new_network, training_options);

save net


