%% FID_test.m —— 按迭代顺序计算 FID（Frechet Inception Distance）并绘制曲线
%  支持两种输入结构:
%    A) 子文件夹模式: iter_500/, iter_1000/, ... 各含多张 PNG（generate_burgers 模式3 输出）
%    B) 单图模式:   扁平目录下按迭代命名的单张/拼贴 PNG（旧版兼容）
%
%  FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrt(Sigma_r * Sigma_g))

clear; clc;

%% ====================== 路径配置 ======================
fprintf('========================================\n');
fprintf('  FID 评分脚本\n');
fprintf('========================================\n');
fprintf('\n期望的文件夹结构（由 generate_burgers.m 模式3 生成）:\n');
fprintf('  您选择的文件夹/\n');
fprintf('    ├── iter_500/       (内含 N 张 img_0001.png ~ img_NNNN.png)\n');
fprintf('    ├── iter_1000/\n');
fprintf('    ├── iter_1500/\n');
fprintf('    └── ...\n');
fprintf('  也支持扁平结构: 文件夹内直接放置按迭代命名的单张/拼贴 PNG。\n\n');

fidDir = uigetdir(pwd, '请选择包含生成图像的文件夹（iter_* 子目录或单张 PNG）');
if fidDir == 0
    error('未选择文件夹。');
end
fprintf('生成图像目录: %s\n', fidDir);

realDir = uigetdir(pwd, '请选择真实图像数据集文件夹（如 Burger_v2）');
if realDir == 0
    error('未选择真实图像文件夹。');
end
fprintf('真实图像目录: %s\n', realDir);

cacheFile = fullfile(fidDir, 'real_features_cache.mat');

%% ====================== 加载 Inception v3 ======================
fprintf('加载 Inception v3 预训练网络...\n');
net = inceptionv3;
dlnet = dag2dlnetwork(net);              % DAGNetwork → dlnetwork，支持指定输出层
inputSize = net.Layers(1).InputSize;     % [299 299 3]
featureLayer = 'avg_pool';               % 2048 维（标准 FID 用法）
fprintf('  输入尺寸: %d x %d x %d\n', inputSize(1), inputSize(2), inputSize(3));
fprintf('  特征层: %s\n', featureLayer);

%% ====================== 提取真实图像特征（带缓存）======================
if exist(cacheFile, 'file')
    fprintf('\n从缓存加载真实图像特征...\n');
    loaded = load(cacheFile, 'mu_real', 'sigma_real', 'numReal');
    mu_real   = loaded.mu_real;
    sigma_real = loaded.sigma_real;
    numReal   = loaded.numReal;
    fprintf('  已加载 %d 张真实图像的特征缓存。\n', numReal);
else
    fprintf('\n提取真实图像特征（首次运行，需要 2-5 分钟，后续走缓存）...\n');
    realFiles = [dir(fullfile(realDir, '*.jpg')); dir(fullfile(realDir, '*.jpeg')); dir(fullfile(realDir, '*.png'))];
    if isempty(realFiles)
        error('未在 %s 中找到图像文件（.jpg / .jpeg / .png）。请检查数据集路径。', realDir);
    end
    numReal = numel(realFiles);
    fprintf('  真实图像数量: %d\n', numReal);

    realFeatures = [];
    batchSize = 32;

    for i = 1:batchSize:numReal
        batchEnd = min(i + batchSize - 1, numReal);
        batchImgs = zeros(inputSize(1), inputSize(2), 3, batchEnd - i + 1, 'single');

        for j = i:batchEnd
            img = imread(fullfile(realDir, realFiles(j).name));
            img = imresize(img, [inputSize(1), inputSize(2)]);
            if size(img, 3) == 1
                img = repmat(img, [1 1 3]);
            end
            batchImgs(:, :, :, j - i + 1) = single(img);   % [0,255]，inceptionv3 内置归一化
        end

        dlBatch = dlarray(batchImgs, 'SSCB');
        if canUseGPU, dlBatch = gpuArray(dlBatch); end

        feats = predict(dlnet, dlBatch, Outputs=featureLayer);
        feats = squeeze(extractdata(gather(feats)))';   % [batchSize x 2048]
        realFeatures = [realFeatures; feats];

        if mod(i - 1, 128) == 0
            fprintf('  进度: %d / %d\n', batchEnd, numReal);
        end
    end
    fprintf('  进度: %d / %d  完成\n', numReal, numReal);

    mu_real = mean(realFeatures, 1);
    sigma_real = cov(realFeatures);
    fprintf('  特征维度: [%d x %d]\n', size(realFeatures, 1), size(realFeatures, 2));

    save(cacheFile, 'mu_real', 'sigma_real', 'numReal');
    fprintf('  特征已缓存至: %s\n', cacheFile);
end

%% ====================== 检测输入结构 ======================
fprintf('\n扫描生成图像目录: %s\n', fidDir);

% 检查是否有 iter_* 子文件夹
iterSubDirs = dir(fullfile(fidDir, 'iter_*'));
iterSubDirs = iterSubDirs([iterSubDirs.isdir]);

if ~isempty(iterSubDirs)
    %% ----- A) 子文件夹模式: 每个 iter_XXXX/ 含多张 PNG -----
    fprintf('  检测到 %d 个迭代子文件夹（多图模式）\n', numel(iterSubDirs));

    % 提取迭代编号并排序
    iterVals = zeros(1, numel(iterSubDirs));
    for i = 1:numel(iterSubDirs)
        tok = regexp(iterSubDirs(i).name, 'iter_(\d+)', 'tokens');
        if ~isempty(tok)
            iterVals(i) = str2double(tok{1}{1});
        else
            iterVals(i) = inf;
        end
    end
    [iterVals, sortIdx] = sort(iterVals, 'ascend');
    iterSubDirs = iterSubDirs(sortIdx);

    % 统计每个子文件夹的图片数
    numCheckpoints = numel(iterSubDirs);
    imgCounts = zeros(1, numCheckpoints);
    for i = 1:numCheckpoints
        subPngs = dir(fullfile(fidDir, iterSubDirs(i).name, '*.png'));
        imgCounts(i) = numel(subPngs);
    end

    fprintf('  检查点数量: %d\n', numCheckpoints);
    fprintf('  每迭代图像数: %d ~ %d\n', min(imgCounts), max(imgCounts));
    for i = 1:numCheckpoints
        fprintf('    [%2d] iter=%5d  %s  (%d 张)\n', ...
            i, iterVals(i), iterSubDirs(i).name, imgCounts(i));
    end

    %% 逐检查点计算 FID
    fprintf('\n开始计算 FID...\n');
    fprintf('----------------------------------------\n');
    fidScores = zeros(1, numCheckpoints);

    for k = 1:numCheckpoints
        iter = iterVals(k);
        subDir = fullfile(fidDir, iterSubDirs(k).name);
        pngFiles = dir(fullfile(subDir, '*.png'));
        numGen = numel(pngFiles);

        fprintf('  [%2d/%2d] iter=%5d (%d 张)...', k, numCheckpoints, iter, numGen);
        tStart = tic;

        % 分批提取特征
        genFeatures = zeros(numGen, numel(mu_real), 'single');
        featBatchSize = 64;

        for j = 1:featBatchSize:numGen
            batchEnd = min(j + featBatchSize - 1, numGen);
            batchImgs = zeros(inputSize(1), inputSize(2), 3, batchEnd - j + 1, 'single');

            for jj = j:batchEnd
                img = imread(fullfile(subDir, pngFiles(jj).name));
                img = imresize(img, [inputSize(1), inputSize(2)]);
                if size(img, 3) == 1
                    img = repmat(img, [1 1 3]);
                end
                batchImgs(:, :, :, jj - j + 1) = single(img);   % [0,255]
            end

            dlBatch = dlarray(batchImgs, 'SSCB');
            if canUseGPU, dlBatch = gpuArray(dlBatch); end
            feats = predict(dlnet, dlBatch, Outputs=featureLayer);
            genFeatures(j:batchEnd, :) = squeeze(extractdata(gather(feats)))';
        end

        % 计算 FID
        mu_gen = mean(genFeatures, 1);
        diff = mu_real - mu_gen;

        if numGen >= 2
            sigma_gen = cov(genFeatures);
            covmean = sqrtm(sigma_real * sigma_gen);
            if ~isreal(covmean)
                covmean = real(covmean);
            end
            fidScores(k) = sum(diff.^2) + trace(sigma_real + sigma_gen - 2 * covmean);
        else
            % 单张退化为特征距离 + 常数项
            fidScores(k) = sum(diff.^2) + trace(sigma_real);
        end

        fprintf(' FID = %8.2f (%.1fs)\n', fidScores(k), toc(tStart));
    end

else
    %% ----- B) 单图模式: 扁平 PNG 文件（向后兼容）-----
    fidFiles = dir(fullfile(fidDir, '*.png'));
    if isempty(fidFiles)
        error('未在 %s 中找到 .png 文件或 iter_* 子文件夹。', fidDir);
    end

    % 从文件名提取迭代次数并排序
    iterVals = zeros(1, numel(fidFiles));
    for i = 1:numel(fidFiles)
        tok = regexp(fidFiles(i).name, '(\d+)', 'tokens');
        if ~isempty(tok)
            iterVals(i) = str2double(tok{1}{1});
        else
            iterVals(i) = inf;
        end
    end
    [iterVals, sortIdx] = sort(iterVals, 'ascend');
    fidFiles = fidFiles(sortIdx);
    numCheckpoints = numel(fidFiles);

    fprintf('  检测到 %d 张单图（兼容模式）\n', numCheckpoints);
    for i = 1:numCheckpoints
        fprintf('    [%2d] iter=%5d  %s\n', i, iterVals(i), fidFiles(i).name);
    end

    fprintf('\n开始计算 FID...\n');
    fprintf('----------------------------------------\n');
    fidScores = zeros(1, numCheckpoints);

    for k = 1:numCheckpoints
        iter  = iterVals(k);
        fname = fidFiles(k).name;
        img   = imread(fullfile(fidDir, fname));

        % 拼贴图检测（边长 ≥500 → 切 5×5）
        [h, w, ~] = size(img);
        if h >= 500 || w >= 500
            gridSize = 5;
            tileH = floor(h / gridSize);
            tileW = floor(w / gridSize);
            numGen = gridSize * gridSize;
            genImgs = zeros(tileH, tileW, 3, numGen, 'uint8');
            idx = 1;
            for r = 1:gridSize
                for c = 1:gridSize
                    rowStart = (r - 1) * tileH + 1;
                    colStart = (c - 1) * tileW + 1;
                    genImgs(:, :, :, idx) = img(rowStart:rowStart + tileH - 1, ...
                                                 colStart:colStart + tileW - 1, :);
                    idx = idx + 1;
                end
            end
        else
            numGen = 1;
            genImgs = img;
        end

        % 提取特征
        genFeatures = zeros(numGen, numel(mu_real), 'single');
        for j = 1:numGen
            imgJ = imresize(genImgs(:, :, :, j), [inputSize(1), inputSize(2)]);
            if size(imgJ, 3) == 1
                imgJ = repmat(imgJ, [1 1 3]);
            end
            dlImg = dlarray(single(imgJ), 'SSCB');
            if canUseGPU, dlImg = gpuArray(dlImg); end
            feat = predict(dlnet, dlImg, Outputs=featureLayer);
            genFeatures(j, :) = squeeze(extractdata(gather(feat)))';
        end

        % 计算 FID
        mu_gen = mean(genFeatures, 1);
        diff = mu_real - mu_gen;

        if numGen >= 2
            sigma_gen = cov(genFeatures);
            covmean = sqrtm(sigma_real * sigma_gen);
            if ~isreal(covmean)
                covmean = real(covmean);
            end
            fidScores(k) = sum(diff.^2) + trace(sigma_real + sigma_gen - 2 * covmean);
        else
            fidScores(k) = sum(diff.^2) + trace(sigma_real);
        end

        fprintf('  [%2d/%2d] iter=%5d | numGen=%2d | FID = %8.2f\n', ...
            k, numCheckpoints, iter, numGen, fidScores(k));
    end
end

fprintf('----------------------------------------\n');

%% ====================== 绘制 FID 曲线（含拟合线）======================
fig = figure('Position', [100 100 960 560], 'Color', 'w');

% 原始数据
plot(iterVals, fidScores, 'b-o', 'LineWidth', 1.8, 'MarkerSize', 8, ...
    'MarkerFaceColor', [0.3 0.6 1], 'MarkerEdgeColor', 'b');
hold on;

% ---- 线性拟合（虚线）----
p_lin = polyfit(iterVals, fidScores, 1);
fid_lin = polyval(p_lin, iterVals);
plot(iterVals, fid_lin, 'r--', 'LineWidth', 1.5);
% 计算 R²
ss_res = sum((fidScores - fid_lin).^2);
ss_tot = sum((fidScores - mean(fidScores)).^2);
r2_lin = 1 - ss_res / ss_tot;

% ---- 对数拟合（点线）----
log_iter = log(iterVals);
p_log = polyfit(log_iter, fidScores, 1);
fid_log = polyval(p_log, log_iter);
plot(iterVals, fid_log, 'm-.', 'LineWidth', 1.5);
% 计算 R²
ss_res2 = sum((fidScores - fid_log).^2);
r2_log = 1 - ss_res2 / ss_tot;

hold off;

xlabel('训练迭代次数', 'FontSize', 13);
ylabel('FID 分数', 'FontSize', 13);
title('FID 随训练迭代变化曲线', 'FontSize', 15);
legend({'FID 实测值', ...
    sprintf('线性拟合  (R^2=%.3f,  斜率=%.3f/千次)', r2_lin, p_lin(1)*1000), ...
    sprintf('对数拟合  (R^2=%.3f)', r2_log)}, ...
    'Location', 'northeast', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 12);

% 标注数值（仅首尾和最低点）
[~, idxMin] = min(fidScores);
labelIdx = unique([1, idxMin, numCheckpoints]);
for k = labelIdx
    text(iterVals(k), fidScores(k), sprintf('%.0f', fidScores(k)), ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', ...
        'FontSize', 9, 'Color', [0.3 0.3 0.3], 'FontWeight', 'bold');
end

%% ====================== 保存结果 ======================
curvePath = fullfile(fidDir, 'FID_curve.png');
saveas(gcf, curvePath);
fprintf('\nFID 曲线图已保存至: %s\n', curvePath);

dataPath = fullfile(fidDir, 'FID_scores.mat');
save(dataPath, 'iterVals', 'fidScores');
fprintf('FID 数据已保存至: %s\n', dataPath);

%% ====================== 汇总输出 ======================
fprintf('\n=============== FID 评分汇总 ===============\n');
fprintf('  %6s   %10s\n', '迭代', 'FID');
fprintf('  ------   ----------\n');
for k = 1:numCheckpoints
    fprintf('  %6d   %10.2f\n', iterVals(k), fidScores(k));
end
fprintf('============================================\n');

fprintf('\n=============== 拟合结果 ===============\n');
fprintf('  线性拟合: FID = %.4f * iter + %.2f\n', p_lin(1), p_lin(2));
fprintf('            R^2 = %.4f,  每千次迭代下降 %.2f\n', r2_lin, -p_lin(1)*1000);
fprintf('  对数拟合: FID = %.2f * ln(iter) + %.2f\n', p_log(1), p_log(2));
fprintf('            R^2 = %.4f\n', r2_log);
fprintf('============================================\n');

fprintf('\n全部完成。\n');
