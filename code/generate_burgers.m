%% 汉堡图像生成脚本 —— 支持三种模式
%  [1] 单一模型 → D 评分筛选 Top-25 → 交互保存
%  [2] 批量模型遍历 → D 评分筛选 Top-25 → 交互保存
%  [3] 批量生成 FID 评估图像（全量保存，不做 D 评分筛选）

clear; clc;

% 添加工具函数路径
scriptDir = fileparts(mfilename('fullpath'));
if ~isempty(scriptDir)
    if exist(fullfile(scriptDir, 'utils'), 'dir')
        addpath(fullfile(scriptDir, 'utils'));
    end
    addpath(scriptDir);
end

%% ==================== 0. 选择运行模式 ====================
fprintf('请选择运行模式:\n');
fprintf('  [1] 运行单一模型（手动选择 .mat 文件）→ D 评分 → Top-25\n');
fprintf('  [2] 遍历文件夹内全部检查点模型 → D 评分 → Top-25\n');
fprintf('  [3] 批量生成 FID 评估图像（遍历文件夹，每个模型生成 N 张，全量保存）\n');
modeChoice = input('请输入 1 / 2 / 3: ', 's');

%% ==================== 0.1 构建文件列表 ====================
if strcmp(modeChoice, '1')
    % 单一模型：手动选择
    [cpFile, cpPath] = uigetfile('*.mat', '选择模型文件');
    if isequal(cpFile, 0)
        error('未选择文件。');
    end
    tok = regexp(cpFile, 'checkpoint_iter_(\d+)\.mat', 'tokens');
    if ~isempty(tok)
        iterNums = str2double(tok{1}{1});
    else
        iterNums = 0;
    end
    cpFiles = struct('name', cpFile);
    sortIdx = 1;
    modelDir = cpPath;

elseif strcmp(modeChoice, '2')
    % 全部模型：用户选择文件夹
    checkpointDir = uigetdir(pwd, '请选择检查点文件夹');
    if checkpointDir == 0
        error('未选择检查点文件夹。');
    end

    cpFiles = dir(fullfile(checkpointDir, '*.mat'));
    if isempty(cpFiles)
        error('未在 %s 中找到 .mat 文件。', checkpointDir);
    end

    iterNums = zeros(1, numel(cpFiles));
    for i = 1:numel(cpFiles)
        tok = regexp(cpFiles(i).name, '(\d+)\.mat', 'tokens');
        if ~isempty(tok)
            iterNums(i) = str2double(tok{1}{1});
        else
            iterNums(i) = -1;
        end
    end
    [iterNums, sortIdx] = sort(iterNums, 'ascend');
    cpFiles = cpFiles(sortIdx);
    modelDir = checkpointDir;

elseif strcmp(modeChoice, '3')
    % === 模式 3: 批量生成 FID 评估图像 ===
    checkpointDir = uigetdir(pwd, '请选择检查点文件夹');
    if checkpointDir == 0
        error('未选择检查点文件夹。');
    end

    cpFiles = dir(fullfile(checkpointDir, '*.mat'));
    if isempty(cpFiles)
        error('未在 %s 中找到 .mat 文件。', checkpointDir);
    end

    iterNums = zeros(1, numel(cpFiles));
    for i = 1:numel(cpFiles)
        tok = regexp(cpFiles(i).name, '(\d+)\.mat', 'tokens');
        if ~isempty(tok)
            iterNums(i) = str2double(tok{1}{1});
        else
            iterNums(i) = -1;
        end
    end
    [iterNums, sortIdx] = sort(iterNums, 'ascend');
    cpFiles = cpFiles(sortIdx);
    modelDir = checkpointDir;

    % 用户设定参数
    numPerModel = input('请输入每个模型生成的图像数量（建议 ≥1000）: ');
    if isempty(numPerModel) || numPerModel <= 0
        error('请提供有效的生成数量。');
    end

    fprintf('请在弹出的对话框中选择输出根目录（FID 图像将保存至 输出目录/iter_XXXX/）...\n');
    outputBaseDir = uigetdir(pwd, '请选择输出根目录（每个迭代将创建子文件夹）');
    if outputBaseDir == 0
        error('未选择输出目录。');
    end

    genBatchSize = min(200, numPerModel);   % 每次 predict 的图像数，避免显存溢出

    %% 循环处理
    fprintf('\n将处理 %d 个模型，每个生成 %d 张图像:\n', numel(cpFiles), numPerModel);
    for i = 1:numel(cpFiles)
        fprintf('  [%2d] iter=%d  %s\n', i, iterNums(i), cpFiles(i).name);
    end
    fprintf('\n');

    totalStart = tic;

    for cpIdx = 1:numel(cpFiles)
        cpPath = fullfile(modelDir, cpFiles(cpIdx).name);
        iter   = iterNums(cpIdx);

        fprintf('========== [%d/%d] iter=%d: %s ==========\n', ...
            cpIdx, numel(cpFiles), iter, cpFiles(cpIdx).name);

        % 加载模型
        loaded = load(cpPath, 'netG', 'netD', 'netG_ema', 'iteration');
        netG = loaded.netG;
        fprintf('  模型已加载 (iter=%d)\n', loaded.iteration);

        if ~exist('numLatentInputs', 'var')
            numLatentInputs = netG.Layers(1).InputSize(3);
            fprintf('  潜在向量维度: %d\n', numLatentInputs);
        end

        % 创建输出子文件夹
        outSubDir = fullfile(outputBaseDir, sprintf('iter_%d', iter));
        if ~exist(outSubDir, 'dir')
            mkdir(outSubDir);
        end

        % 分批生成并保存
        numBatches = ceil(numPerModel / genBatchSize);
        imgIdx = 1;
        batchStart = tic;

        for b = 1:numBatches
            currentBatch = min(genBatchSize, numPerModel - (b - 1) * genBatchSize);
            fprintf('  批次 %d/%d: 生成 %d 张...', b, numBatches, currentBatch);

            ZNew = randn(1, 1, numLatentInputs, currentBatch, 'single');
            ZNew = dlarray(ZNew, 'SSCB');
            if canUseGPU, ZNew = gpuArray(ZNew); end

            XNew = predict(netG, ZNew);
            batchImgs = gather(extractdata(XNew));   % [128 128 3 currentBatch] single, [-1, 1]

            % 转换并保存: tanh [-1,1] → [0,1] → uint8
            batchImgs = (batchImgs + 1) / 2;
            batchImgs = max(0, min(1, batchImgs));

            for j = 1:currentBatch
                fname = fullfile(outSubDir, sprintf('img_%04d.png', imgIdx));
                imwrite(im2uint8(batchImgs(:, :, :, j)), fname);
                imgIdx = imgIdx + 1;
            end

            fprintf(' 完成 (%.1fs)\n', toc(batchStart));
        end

        elapsed = toc(totalStart);
        fprintf('  → 已保存 %d 张图像到: %s (累计耗时 %.1fs)\n\n', ...
            numPerModel, outSubDir, elapsed);
    end

    fprintf('全部处理完毕。总耗时 %.1f 分钟。\n', toc(totalStart) / 60);
    return;   % 模式 3 结束，不进入后面的 D 评分流程

else
    error('无效选择: %s。请输入 1、2 或 3。', modeChoice);
end

%% ==================== 模式 1 & 2: 显示文件列表 ====================
fprintf('\n将处理 %d 个模型:\n', numel(cpFiles));
for i = 1:numel(cpFiles)
    fprintf('  [%2d] iter=%d  %s\n', i, iterNums(i), cpFiles(i).name);
end
fprintf('\n');

%% ==================== 1. 逐个检查点处理 (D 评分 + 交互保存) ====================
numCandidates = 200;
numTop        = 25;

for cpIdx = 1:numel(cpFiles)
    cpPath = fullfile(modelDir, cpFiles(cpIdx).name);
    iter   = iterNums(cpIdx);

    fprintf('========== [%d/%d] 加载: %s ==========\n', ...
        cpIdx, numel(cpFiles), cpFiles(cpIdx).name);

    % 加载检查点
    loaded = load(cpPath, 'netG', 'netD', 'netG_ema', 'iteration');
    netG = loaded.netG;
    netD = loaded.netD;
    if isfield(loaded, 'netG_ema')
        netG_ema = loaded.netG_ema;  %#ok<NASGU>
    end
    checkpointIter = loaded.iteration;

    if ~exist('numLatentInputs', 'var')
        numLatentInputs = netG.Layers(1).InputSize(3);
    end

    %% 生成候选图像
    fprintf('  生成 %d 张候选图像...\n', numCandidates);
    ZNew = randn(1, 1, numLatentInputs, numCandidates, 'single');
    ZNew = dlarray(ZNew, 'SSCB');
    if canUseGPU, ZNew = gpuArray(ZNew); end

    XNew = predict(netG, ZNew);
    allImgs = gather(extractdata(XNew));   % [128 128 3 200] single, CPU

    %% D 判别器评分
    fprintf('  D 判别器评分中...\n');
    XForD = dlarray(single(allImgs), 'SSCB');
    if canUseGPU, XForD = gpuArray(XForD); end

    out = predict(netD, XForD);
    scores = squeeze(extractdata(gather(sigmoid(out))))';   % [200×1]

    [sortedScores, sortedIdx] = sort(scores, 'descend');
    topIdx = sortedIdx(1:numTop);

    %% 显示结果
    bestImgs = allImgs(:, :, :, topIdx);
    Inew = imtile(bestImgs);
    Inew = rescale(Inew);

    figure('Name', sprintf('Iter %d — D 评分 Top-%d', checkpointIter, numTop));
    image(Inew); axis off;
    title(sprintf('Iter %d | Top-1: %.4f | Top-25: %.4f', ...
        checkpointIter, sortedScores(1), sortedScores(numTop)));
    drawnow;

    fprintf('  Top-5 分数: ');
    for i = 1:5
        fprintf('%.4f  ', sortedScores(i));
    end
    fprintf('\n');

    %% 询问保存
    fprintf('\n  --- 保存选项 ---\n');
    saveChoice = input('  保存？ [1]是 / [2]否 / [0]退出: ', 's');

    if strcmp(saveChoice, '0')
        fprintf('  已退出。\n');
        break;
    elseif strcmp(saveChoice, '2')
        fprintf('  跳过此检查点。\n\n');
        continue;
    end

    % 选择保存方式
    fprintf('  保存方式: [1]仅最高分一张 / [2]全部 %d 张: ', numTop);
    saveMode = input('', 's');

    if strcmp(saveMode, '1')
        [saveFile, savePath] = uiputfile( ...
            {'*.png', 'PNG 图像 (*.png)'}, ...
            sprintf('保存 Iter %d 最佳图像', checkpointIter));
        if isequal(saveFile, 0)
            fprintf('  未选择保存路径，跳过。\n\n');
            continue;
        end

        bestImg = bestImgs(:, :, :, 1);
        bestImg = rescale(bestImg);
        bestImg_uint8 = im2uint8(bestImg);
        imwrite(bestImg_uint8, fullfile(savePath, saveFile));
        fprintf('  已保存最佳图像到: %s\n', fullfile(savePath, saveFile));

    else
        saveDir = uigetdir(pwd, sprintf('选择保存文件夹 (Iter %d)', checkpointIter));
        if saveDir == 0
            fprintf('  未选择保存路径，跳过。\n\n');
            continue;
        end

        subDir = fullfile(saveDir, sprintf('iter_%d', checkpointIter));
        mkdir(subDir);

        for i = 1:numTop
            img_i = bestImgs(:, :, :, i);
            img_i = rescale(img_i);
            img_i_uint8 = im2uint8(img_i);
            fileName = fullfile(subDir, sprintf('rank%02d_score%.4f.png', ...
                i, sortedScores(i)));
            imwrite(img_i_uint8, fileName);
        end

        montageFile = fullfile(subDir, sprintf('montage_iter%d.png', checkpointIter));
        imwrite(im2uint8(Inew), montageFile);

        scoreFile = fullfile(subDir, 'scores.txt');
        fid = fopen(scoreFile, 'w');
        fprintf(fid, 'Rank\tScore\n');
        for i = 1:numTop
            fprintf(fid, '%d\t%.6f\n', i, sortedScores(i));
        end
        fclose(fid);

        fprintf('  已保存 %d 张 + 拼贴图 + 评分记录到:\n    %s\n', numTop, subDir);
    end

    fprintf('\n');
end

fprintf('全部处理完毕。\n');
