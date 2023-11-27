[predicted_label, probability] = classify(net, resized_testing_dataset);
accuracy= mean(predicted_label == TestingDataset.Labels);
index = randperm(numel(resized_testing_dataset.Files), 4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(TestingDataset, index(i));
    imshow(I)
    label = predicted_label(index(i));
    title(string(label)+ ", " + num2str(max(probability(index(i), :)), 3) +'%');
end