%% 加载 DCGAN 模型，批量生成 200 张 → D 评分筛选 → 保留 Top-25
clear; clc;

% 添加工具函数路径
scriptDir = fileparts(mfilename('fullpath'));
if ~isempty(scriptDir)
    addpath(fullfile(scriptDir, 'utils'));
end

% 加载模型（含 netG 和 netD）
modelFile = fullfile(scriptDir, '..', 'model', 'hamburger_gan_final.mat');
if ~isfile(modelFile)
    % 回退：尝试当前目录
    modelFile = 'hamburger_gan_final.mat';
end
load(modelFile);

% 推断潜在向量维度
if ~exist('numLatentInputs', 'var')
    numLatentInputs = netG.Layers(1).InputSize(3);
end

%% 1. 批量生成 200 张候选图像
numCandidates = 200;
numTop        = 25;

fprintf('正在生成 %d 张候选图像...\n', numCandidates);

ZNew = randn(1, 1, numLatentInputs, numCandidates, 'single');
ZNew = dlarray(ZNew, 'SSCB');
if canUseGPU
    ZNew = gpuArray(ZNew);
end

XNew = predict(netG, ZNew);

% ★ 关键：立刻把全部生成结果搬到 CPU 纯数组
allImgs = gather(extractdata(XNew));   % [128 128 3 200] single, CPU

%% 2. 判别器批量评分
fprintf('正在用判别器评分...\n');

% 整个 batch 一次性送进 D（SSCB 四标签匹配）
XForD = dlarray(single(allImgs), 'SSCB');
if canUseGPU
    XForD = gpuArray(XForD);
end

out = predict(netD, XForD);
scores = squeeze(extractdata(gather(sigmoid(out))))';   % [200×1] 概率分数

% 按分数降序排列，取 Top-25
[~, sortedIdx] = sort(scores, 'descend');
topIdx = sortedIdx(1:numTop);

fprintf('评分完成。最高分: %.4f | 最低分: %.4f | Top-25 阈值: %.4f\n', ...
    max(scores), min(scores), scores(topIdx(end)));

%% 3. 筛选最佳图像（纯数组索引，无 dlarray 问题）
bestImgs = allImgs(:, :, :, topIdx);

% 拼贴展示
Inew = imtile(bestImgs);
Inew = rescale(Inew);
figure('Name', sprintf('D 评分筛选 Top-%d', numTop));
image(Inew); axis off;
title(sprintf('DCGAN 生成汉堡（%d 候选 → D 评分 → Top-%d）', numCandidates, numTop));
drawnow;

%% 4. 显示分数排名
fprintf('\n========== Top-%d 评分 ==========\n', numTop);
for i = 1:numTop
    fprintf('  %2d. 候选#%03d  分数: %.4f\n', i, topIdx(i), scores(topIdx(i)));
end

%% 5. 保存图像
if isdeployed || ~usejava('desktop')
    % 非交互模式：自动保存到当前目录
    timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    saveSubDir = fullfile(pwd, ['generated_burgers_Dscored_', timestamp]);
    mkdir(saveSubDir);
else
    answer = questdlg('是否保存筛选后的图像？', '保存图像', ...
        '是，选择保存路径', '否', '是，选择保存路径');
    if strcmp(answer, '是，选择保存路径')
        saveDir = uigetdir(pwd, '选择保存文件夹');
        if saveDir == 0
            warning('未选择文件夹，图像未保存。');
            return;
        end
        timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
        saveSubDir = fullfile(saveDir, ['generated_burgers_Dscored_', timestamp]);
        mkdir(saveSubDir);
    else
        disp('图像未保存。');
        return;
    end
end

for i = 1:numTop
    img_i = bestImgs(:, :, :, i);
    img_i = rescale(img_i);
    img_i_uint8 = im2uint8(img_i);
    fileName = fullfile(saveSubDir, sprintf('burger_%02d_score%.4f.png', ...
        i, scores(topIdx(i))));
    imwrite(img_i_uint8, fileName);
end

% 保存拼贴图
montageFile = fullfile(saveSubDir, 'montage_5x5_Dscored.png');
imwrite(im2uint8(Inew), montageFile);

% 保存评分记录
scoreFile = fullfile(saveSubDir, 'scores.txt');
fid = fopen(scoreFile, 'w');
fprintf(fid, 'Rank\tCandidate#\tScore\n');
for i = 1:numTop
    fprintf(fid, '%d\t%d\t%.6f\n', i, topIdx(i), scores(topIdx(i)));
end
fclose(fid);

fprintf('已保存 %d 张单图 + 拼贴图 + 评分记录到:\n   %s\n', numTop, saveSubDir);

