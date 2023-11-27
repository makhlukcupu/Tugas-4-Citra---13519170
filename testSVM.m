testImage = readimage(TestingDataset, 12);
scaleTestImage = imresize(testImage, [256, 256]);
featureTest = extractHOGFeatures(scaleTestImage);
[predictIndex, score] = predict(SVM_classifier, featureTest);
figure; imshow(testImage); title(char(predictIndex));