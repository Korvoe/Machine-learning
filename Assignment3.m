mnist_dataset = load('mnist.mat');
trdata = mnist_dataset.trdata;
trlabels = mnist_dataset.trlabels;
tsdata = mnist_dataset.trdata;
tslabels = mnist_dataset.tslabels;

%Feature extraction

%GLCM
glcm_energy = zeros(2, 3495);
for i = 1:3495
    glcm = graycoprops(graycomatrix(trdata(1:28,1:28,1,i),'NumLevels',10,'GrayLimits',[],'Offset',[1 0]),{'Energy'});
    glcm_energy(1,i) = glcm.Energy;
    glcm = graycoprops(graycomatrix(trdata(1:28,1:28,1,i),'NumLevels',10,'GrayLimits',[],'Offset',[0 1]),{'Energy'});
    glcm_energy(2,i) = glcm.Energy;
end

%Gabor
gabor_energy = zeros(6, 3495);
for i = 1:3495
    gb = imgaborfilt(trdata(1:28,1:28,1,i), gabor([4 8], [0 45 90]));
    [m,n,~]= size(gb);
    for j = 1:6
        gabor_energy(j,i) = sum(reshape(gb(:,:,j),1,[]).^2)./(m*n);
    end
end

%LBP
mapping1 = getmapping(8,'u2');
mapping2 = getmapping(16,'u2');
hist1 = zeros(1, 59, 3495);
hist2 = zeros(1, 243, 3495);
for i = 1:3495
    lbp1 = lbp(trdata(:,:,1,i), 1, 8, mapping1, 'i');
    lbp2 = lbp(trdata(:,:,1,i), 2, 16, mapping2, 'i');
    hist1(:,:,i) = hist(lbp1(:), 0:(mapping1.num-1));
    hist2(:,:,i) = hist(lbp2(:), 0:(mapping2.num-1));
    hist1(:,:,i) = hist1(:,:,i)./sum(hist1(:,:,i));
    hist2(:,:,i) = hist2(:,:,i)./sum(hist2(:,:,i));
end
hist1 = squeeze(hist1);
hist2 = squeeze(hist2);



%Max`s and min`s
glcm_max = zeros(1,2);
glcm_min = zeros(1,2);
for i = 1:2
    glcm_max(i) = max(glcm_energy(i,:));
    glcm_min(i) = min(glcm_energy(i,:));
end

gabor_max = zeros(1,6);
gabor_min = zeros(1,6);
for i = 1:6
    gabor_max(i) = max(gabor_energy(i,:));
    gabor_min(i) = min(gabor_energy(i,:));
end

lbp_8_max = max(hist1(:));
lbp_8_min = min(hist1(:));
lbp_16_max = max(hist2(:));
lbp_16_min = min(hist2(:));



%Normalization
normalized_glcm = zeros(2,3495);
for i = 1:2
   for j = 1:3495
        normalized_glcm(i,j) = (glcm_energy(i,j) - glcm_min(i))/(glcm_max(i) - glcm_min(i));
    end
end

normalized_gabor = zeros(6, 3495);
for i = 1:6
    for j = 1:3495
        normalized_gabor(i,j) = (gabor_energy(i,j) - gabor_min(i))/(gabor_max(i) - gabor_min(i));
    end
end

normalized_8_lbp = zeros(59,3495);
for i = 1:59
    for j = 1:3495
        normalized_8_lbp(i,j) = (hist1(i,j) - lbp_8_min)/(lbp_8_max - lbp_8_min);
    end
end

normalized_16_lbp = zeros(243,3495);
for i = 1:243
    for j = 1:3495
        normalized_16_lbp(i,j) = (hist2(i,j) - lbp_16_min)/(lbp_16_max - lbp_16_min);
    end
end

%Training
glcm_model = fitcnb(normalized_glcm.', trlabels);
gabor_model = fitcnb(normalized_gabor.', trlabels);
lbp_model = fitcnb([normalized_8_lbp; normalized_16_lbp].', trlabels);
glcm_gabor_model = fitcnb([normalized_glcm; normalized_gabor].', trlabels);
glcm_lbp_model = fitcnb([normalized_glcm; normalized_8_lbp; normalized_16_lbp].', trlabels);
gabor_lbp_model = fitcnb([normalized_gabor; normalized_8_lbp; normalized_16_lbp].', trlabels);
glcm_gabor_lbp_model = fitcnb([normalized_glcm; normalized_gabor; normalized_8_lbp; normalized_16_lbp].', trlabels);




%Test data feature extraction

%GLCM
ts_glcm = zeros(2, 3495);
for i = 1:3495
    glcm = graycoprops(graycomatrix(tsdata(1:28,1:28,1,i),'NumLevels',10,'GrayLimits',[],'Offset',[1 0]),{'Energy'});
    ts_glcm(1,i) = glcm.Energy;
    glcm = graycoprops(graycomatrix(tsdata(1:28,1:28,1,i),'NumLevels',10,'GrayLimits',[],'Offset',[0 1]),{'Energy'});
    ts_glcm(2,i) = glcm.Energy;
end

%Gabor
ts_gabor = zeros(6, 3495);
for i = 1:3495
    gb = imgaborfilt(tsdata(1:28,1:28,1,i), gabor([4 8], [0 45 90]));
    [m,n,~]= size(gb);
    for j = 1:6
        ts_gabor(j,i) = sum(reshape(gb(:,:,j),1,[]).^2)./(m*n);
    end
end

%LBP
mapping1 = getmapping(8,'u2');
mapping2 = getmapping(16,'u2');
ts_hist1 = zeros(1, 59, 3495);
ts_hist2 = zeros(1, 243, 3495);
for i = 1:3495
    lbp1 = lbp(tsdata(:,:,1,i), 1, 8, mapping1, 'i');
    lbp2 = lbp(tsdata(:,:,1,i), 2, 16, mapping2, 'i');
    ts_hist1(:,:,i) = hist(lbp1(:), 0:(mapping1.num-1));
    ts_hist2(:,:,i) = hist(lbp2(:), 0:(mapping2.num-1));
    ts_hist1(:,:,i) = ts_hist1(:,:,i)./sum(ts_hist1(:,:,i));
    ts_hist2(:,:,i) = ts_hist2(:,:,i)./sum(ts_hist2(:,:,i));
end
ts_hist1 = squeeze(ts_hist1);
ts_hist2 = squeeze(ts_hist2);

%Max`s and min`s
glcm_max = zeros(1,2);
glcm_min = zeros(1,2);
for i = 1:2
    glcm_max(i) = max(ts_glcm(i,:));
    glcm_min(i) = min(ts_glcm(i,:));
end

gabor_max = zeros(1,6);
gabor_min = zeros(1,6);
for i = 1:6
    gabor_max(i) = max(ts_gabor(i,:));
    gabor_min(i) = min(ts_gabor(i,:));
end

lbp_8_max = max(ts_hist1(:));
lbp_8_min = min(ts_hist1(:));
lbp_16_max = max(ts_hist2(:));
lbp_16_min = min(ts_hist2(:));



%Normalization
normalized_ts_glcm = zeros(2,3495);
for i = 1:2
   for j = 1:3495
        normalized_ts_glcm(i,j) = (ts_glcm(i,j) - glcm_min(i))/(glcm_max(i) - glcm_min(i));
    end
end

normalized_ts_gabor = zeros(6, 3495);
for i = 1:6
    for j = 1:3495
        normalized_ts_gabor(i,j) = (ts_gabor(i,j) - gabor_min(i))/(gabor_max(i) - gabor_min(i));
    end
end

normalized_ts_8_lbp = zeros(59,3495);
for i = 1:59
    for j = 1:3495
        normalized_ts_8_lbp(i,j) = (ts_hist1(i,j) - lbp_8_min)/(lbp_8_max - lbp_8_min);
    end
end

normalized_ts_16_lbp = zeros(243,3495);
for i = 1:243
    for j = 1:3495
        normalized_ts_16_lbp(i,j) = (ts_hist2(i,j) - lbp_16_min)/(lbp_16_max - lbp_16_min);
    end
end



%Prediction on training data 
tr_glcm_predicted_labels = predict(glcm_model, normalized_glcm.').';
tr_gabor_predicted_labels = predict(gabor_model, normalized_gabor.').';
tr_LBP_predicted_labels = predict(lbp_model, [normalized_8_lbp; normalized_16_lbp].').';
tr_glcm_gabor_predicted_labels = predict(glcm_gabor_model, [normalized_glcm; normalized_gabor].').';
tr_glcm_LBP_predicted_labels = predict(glcm_lbp_model, [normalized_glcm; normalized_8_lbp; normalized_ts_16_lbp;].').';
tr_gabor_LBP_predicted_labels = predict(gabor_lbp_model, [normalized_gabor; normalized_8_lbp; normalized_16_lbp].').';
tr_glcm_gabor_LBP_predicted_labels = predict(glcm_gabor_lbp_model, [normalized_glcm; normalized_gabor; normalized_8_lbp; normalized_16_lbp].').';
correct_predictions = 0;
for i = 1:3495
    if (tr_glcm_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Training Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : GLCM feature]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (tr_gabor_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Training Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : Gabor feature]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (tr_LBP_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Training Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : LBP features]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (tr_glcm_gabor_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Training Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : GLCM and Gabor features]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (tr_glcm_LBP_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Training Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : GLCM and LBP features]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (tr_gabor_LBP_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Training Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : Gabor and LBP features]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (tr_glcm_gabor_LBP_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Training Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : GLCM, Gabor and LBP feature]");
disp(X);

%Prediction on testing data 
ts_glcm_predicted_labels = predict(glcm_model, normalized_ts_glcm.').';
ts_gabor_predicted_labels = predict(gabor_model, normalized_ts_gabor.').';
ts_LBP_predicted_labels = predict(lbp_model, [normalized_ts_8_lbp; normalized_ts_16_lbp].').';
ts_glcm_gabor_predicted_labels = predict(glcm_gabor_model, [normalized_ts_glcm; normalized_ts_gabor].').';
ts_glcm_LBP_predicted_labels = predict(glcm_lbp_model, [normalized_ts_glcm; normalized_ts_8_lbp; normalized_ts_16_lbp;].').';
ts_gabor_LBP_predicted_labels = predict(gabor_lbp_model, [normalized_ts_gabor; normalized_ts_8_lbp; normalized_ts_16_lbp].').';
ts_glcm_gabor_LBP_predicted_labels = predict(glcm_gabor_lbp_model, [normalized_ts_glcm; normalized_ts_gabor; normalized_ts_8_lbp; normalized_ts_16_lbp].').';
correct_predictions = 0;
for i = 1:3495
    if (ts_glcm_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Testing Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : GLCM feature]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (ts_gabor_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Testing Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : Gabor feature]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (ts_LBP_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Testing Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : LBP features]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (ts_glcm_gabor_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Testing Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : GLCM and Gabor features]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (ts_glcm_LBP_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Testing Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : GLCM and LBP features]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (ts_gabor_LBP_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Testing Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : Gabor and LBP features]");
disp(X);

correct_predictions = 0;
for i = 1:3495
    if (ts_glcm_gabor_LBP_predicted_labels(i) == trlabels(i))
        correct_predictions = correct_predictions+1;
    end
end
X = sprintf('Overall Testing Accuracy: %f%%', correct_predictions/3495);
disp("[Classification : GLCM, Gabor and LBP feature]");
disp(X);
