%% DCGAN 汉堡图像生成（128×128，LSGAN + Feature Matching + Minibatch StdDev + EMA）
% 数据集：Burger_v2（963 张纯汉堡裁剪 + 白底填充 256×256）
% 输出：128×128×3 RGB 汉堡图像
% 显存需求：~5.5 GB（RTX 4060 8GB 安全）

clear; clc; close all;

%% ========== 0. 随机种子与训练模式 ==========
rng(42);
if canUseGPU
    try
        gpurng(42);
    catch
    end
end

fprintf('============================================\n');
fprintf('  DCGAN 汉堡生成训练\n');
fprintf('  架构: 转置卷积 (kernel=4) | 128×128\n');
fprintf('  损失: LSGAN + λFM=8 (单层 disc_feat) + Minibatch StdDev\n');
fprintf('  策略: EMA 生成器 | D/G Adam β₁=0.0/0.5 | 余弦退火\n');
fprintf('  数据集: Burger_v2（963 张纯汉堡 + 白底填充）\n');
fprintf('============================================\n\n');

fprintf('请选择训练模式:\n');
fprintf('  [1] 从零开始训练\n');
fprintf('  [2] 从检查点续训\n');
mode = input('请输入 1 或 2: ');

checkpointDir = fullfile(pwd, 'checkpoints_gan');
if ~isfolder(checkpointDir)
    mkdir(checkpointDir);
end

if mode == 2
    [cpFile, cpPath] = uigetfile('*.mat', '选择检查点文件', ...
        fullfile(checkpointDir, '*.mat'));
    if isequal(cpFile, 0)
        error('未选择检查点文件，训练终止。');
    end
    cpFullPath = fullfile(cpPath, cpFile);
    fprintf('加载检查点: %s\n', cpFullPath);
    loaded = load(cpFullPath, 'netG','netD','netG_ema','iteration','epoch', ...
        'trailingAvgG','trailingAvgSqG','trailingAvgD','trailingAvgSqD');
    fprintf('  检查点: Epoch=%d, Iteration=%d\n', loaded.epoch, loaded.iteration);
    startEpoch     = loaded.epoch;
    startIter      = loaded.iteration;
    netG           = loaded.netG;
    netD           = loaded.netD;
    if isfield(loaded, 'netG_ema')
        netG_ema   = loaded.netG_ema;
    else
        netG_ema   = netG;
    end
    trailingAvgG   = loaded.trailingAvgG;
    trailingAvgSqG = loaded.trailingAvgSqG;
    trailingAvgD   = loaded.trailingAvgD;
    trailingAvgSqD = loaded.trailingAvgSqD;
else
    fprintf('从零开始训练。\n');
    startEpoch     = 0;
    startIter      = 0;
    trailingAvgG   = [];
    trailingAvgSqG = [];
    trailingAvgD   = [];
    trailingAvgSqD = [];
end

%% Minibatch 标准差层（内联定义）
function layer = minibatchStddevLayer(layerName)
    fcn = @(X) minibatchStddevFcn(X);
    layer = functionLayer(fcn, 'Name', layerName, 'Formattable', true);
end

function Y = minibatchStddevFcn(X)
    mu    = mean(X, 4);
    s     = sqrt(mean((X - mu).^2, 4) + 1e-8);
    s_avg = mean(s, 'all');
    s_map = repmat(s_avg, [size(X,1), size(X,2), 1, size(X,4)]);
    Y     = cat(3, X, s_map);
end

%% 1. 加载数据
imageFolder = 'D:\hamberger_datasets\Fast Food Classification V2\Train\Burger_v2';
if ~isfolder(imageFolder)
    error('找不到文件夹，请检查路径： %s', imageFolder);
end

imds = imageDatastore(imageFolder, 'IncludeSubfolders', true);
fprintf('共找到 %d 张汉堡图片\n', numel(imds.Files));

if mode == 1
    figure('Name','Burger_v2 数据集样本');
    sampleImages = imds.readall();
    numShow = min(16, numel(sampleImages));
    montage(sampleImages(1:numShow), 'Size', [4 4]);
    title('Burger_v2 —— 纯汉堡主体 + 白底填充');
end

%% 2. 数据增强与预处理
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandScale', [0.9, 1.1], ...
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);
augimds = augmentedImageDatastore([128 128], imds, 'DataAugmentation', augmenter);

%% 3. 生成器网络（转置卷积 kernel=4，均匀重叠无棋盘格）
if mode == 1
    numLatentInputs = 128;

    layersGenerator = [
        imageInputLayer([1 1 numLatentInputs], 'Normalization','none', 'Name','noise')

        % 1×1 → 4×4×512
        transposedConv2dLayer(4, 512, 'Stride',1, 'Cropping',0, 'Name','tconv_g1')
        batchNormalizationLayer('Name','bn_g1')
        reluLayer('Name','relu_g1')

        % 4×4 → 8×8×256
        transposedConv2dLayer(4, 256, 'Stride',2, 'Cropping','same', 'Name','tconv_g2')
        batchNormalizationLayer('Name','bn_g2')
        reluLayer('Name','relu_g2')

        % 8×8 → 16×16×128
        transposedConv2dLayer(4, 128, 'Stride',2, 'Cropping','same', 'Name','tconv_g3')
        batchNormalizationLayer('Name','bn_g3')
        reluLayer('Name','relu_g3')

        % 16×16 → 32×32×64
        transposedConv2dLayer(4, 64, 'Stride',2, 'Cropping','same', 'Name','tconv_g4')
        batchNormalizationLayer('Name','bn_g4')
        reluLayer('Name','relu_g4')

        % 32×32 → 64×64×32
        transposedConv2dLayer(4, 32, 'Stride',2, 'Cropping','same', 'Name','tconv_g5')
        batchNormalizationLayer('Name','bn_g5')
        reluLayer('Name','relu_g5')

        % 64×64 → 128×128×16
        transposedConv2dLayer(4, 16, 'Stride',2, 'Cropping','same', 'Name','tconv_g6')
        batchNormalizationLayer('Name','bn_g6')
        reluLayer('Name','relu_g6')

        % 输出：128×128×3
        transposedConv2dLayer(4, 3, 'Stride',1, 'Cropping','same', 'Name','tconv_out')
        tanhLayer('Name','tanh_out')
    ];

    netG = dlnetwork(layersGenerator);
    fprintf('生成器构建成功（转置卷积 kernel=4，%d 维噪声 → 128×128×3）。\n', numLatentInputs);
else
    numLatentInputs = netG.Layers(1).InputSize(3);
end

%% 4. 判别器网络
if mode == 1
    dropoutProb = 0.3;
    lreluScale  = 0.2;

    layersDiscriminator = [
        imageInputLayer([128 128 3], 'Normalization','none', 'Name','input')
        dropoutLayer(dropoutProb, 'Name','dropout_in')

        % 128 → 64, ch=64
        convolution2dLayer(4, 64, 'Stride',2, 'Padding','same', 'Name','conv_d1')
        leakyReluLayer(lreluScale, 'Name','lrelu_d1')

        % 64 → 32, ch=128
        convolution2dLayer(4, 128, 'Stride',2, 'Padding','same', 'Name','conv_d2')
        batchNormalizationLayer('Name','bn_d2')
        leakyReluLayer(lreluScale, 'Name','lrelu_d2')

        % 32 → 16, ch=256
        convolution2dLayer(4, 256, 'Stride',2, 'Padding','same', 'Name','conv_d3')
        batchNormalizationLayer('Name','bn_d3')
        leakyReluLayer(lreluScale, 'Name','lrelu_d3')

        % 16 → 8, ch=256
        convolution2dLayer(4, 256, 'Stride',2, 'Padding','same', 'Name','conv_d4')
        batchNormalizationLayer('Name','bn_d4')
        leakyReluLayer(lreluScale, 'Name','lrelu_d4')

        % 8 → 4, ch=256 — 特征匹配层
        convolution2dLayer(4, 256, 'Stride',2, 'Padding','same', 'Name','conv_d5')
        batchNormalizationLayer('Name','bn_d5')
        leakyReluLayer(lreluScale, 'Name','disc_feat')

        % Minibatch 标准差层
        minibatchStddevLayer('minibatch_stddev')

        % 4×4×257 → 输出
        convolution2dLayer(4, 1, 'Name','output')
    ];

    netD = dlnetwork(layersDiscriminator);
    fprintf('判别器构建成功（kernel=4，含 Minibatch StdDev 层）。\n\n');
end

%% 5. 训练超参数
numEpochs     = 300;
miniBatchSize = 32;

learnRateD_initial = 3e-5;
learnRateG_initial = 1e-4;
learnRateD_final   = 5e-7;
learnRateG_final   = 5e-7;

gradientDecayFactor_D        = 0.0;
gradientDecayFactor_G        = 0.5;
squaredGradientDecayFactor   = 0.999;

validationFrequency = 50;

lambdaFM      = 8;
featLayerName = 'disc_feat';

maxGradNormD  = 50;
maxGradNormG  = 25;

checkpointFrequency  = 500;
maxCheckpoints       = 5;
anomalyWarmup        = 3000;
anomalyMeanThreshold = 0.25;
anomalyStdMin        = 0.008;

%% 6. 设置 minibatchqueue
augimds.MiniBatchSize = miniBatchSize;
mbq = minibatchqueue(augimds, ...
    'MiniBatchSize', miniBatchSize, ...
    'PartialMiniBatch', 'discard', ...
    'MiniBatchFcn', @preprocessMiniBatch, ...
    'MiniBatchFormat', 'SSCB');

%% 7. 初始化 EMA 生成器
if mode == 1
    netG_ema = netG;
end
emaBeta = 0.999;

%% 8. 初始化监视器与验证噪声
numValidationImages = 25;
ZValidation = randn(1, 1, numLatentInputs, numValidationImages, 'single');
ZValidation = dlarray(ZValidation, 'SSCB');

if canUseGPU
    ZValidation = gpuArray(ZValidation);
    disp('检测到 GPU，将使用 GPU 训练。');
else
    disp('未检测到 GPU，将使用 CPU 训练。');
end

numObservations       = numel(imds.Files);
numIterationsPerEpoch = floor(numObservations / miniBatchSize);
numIterations         = numEpochs * numIterationsPerEpoch;

monitor = createMonitor();

%% 9. 初始化图像监控窗口
hFig = figure('Name','生成图像监控','NumberTitle','off');
hAx = axes(hFig);
hIm = imshow(zeros(5*128, 5*128, 3), 'Parent', hAx);
title(hAx, '初始化...');
drawnow;

genStatsHistory.mean       = [];
genStatsHistory.std        = [];
genStatsHistory.windowSize = 5;

bestScoreG    = 0;
bestIteration = 0;
crashCount    = 0;
maxRecoveries = 3;

nanFMCounter   = 0;
nanFMMaxSilent = 50;

%% 10. 训练循环
epoch     = startEpoch;
iteration = startIter;
lastFMLoss = single(0);

fprintf('\n========== 开始训练（从 Epoch %d / Iter %d） ==========\n', epoch, iteration);
fprintf('数据集: Burger_v2（%d 张纯汉堡裁剪 + 白底填充）\n', numObservations);
fprintf('架构: 转置卷积 kernel=4 | 128×128 | Batch: %d\n', miniBatchSize);
fprintf('λFM=%.0f (单层 disc_feat) | Minibatch StdDev: ON | EMA: β=%.3f\n', lambdaFM, emaBeta);
fprintf('Adam β₁: D=%.1f, G=%.1f | β₂=%.3f\n', ...
    gradientDecayFactor_D, gradientDecayFactor_G, squaredGradientDecayFactor);
fprintf('初始 LR_D: %.2e | 初始 LR_G: %.2e | 余弦退火至 %.0e\n', ...
    learnRateD_initial, learnRateG_initial, learnRateG_final);
fprintf('每 Epoch: %d 步 | 总迭代: %d（%d epoch × %d 步/epoch）\n', ...
    numIterationsPerEpoch, numIterations, numEpochs, numIterationsPerEpoch);
fprintf('崩溃检测: anomalyWarmup=%d, stdMin=%.3f, meanThreshold=%.2f\n', ...
    anomalyWarmup, anomalyStdMin, anomalyMeanThreshold);
fprintf('检查点目录: %s（保留最近 %d 个）\n\n', checkpointDir, maxCheckpoints);

while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;
    shuffle(mbq);

    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;

        progress = iteration / numIterations;
        learnRateD = learnRateD_final + 0.5 * (learnRateD_initial - learnRateD_final) * ...
            (1 + cos(pi * progress));
        learnRateG = learnRateG_final + 0.5 * (learnRateG_initial - learnRateG_final) * ...
            (1 + cos(pi * progress));

        X = next(mbq);
        Z = randn(1, 1, numLatentInputs, miniBatchSize, 'single');
        Z = dlarray(Z, 'SSCB');
        if canUseGPU, Z = gpuArray(Z); end

        [~, ~, gradientsG, gradientsD, stateG, scoreG, scoreD, lossFM] = ...
            dlfeval(@modelLoss, netG, netD, X, Z, featLayerName, lambdaFM);

        % NaN/Inf 梯度守卫
        if gradsHaveNaN(gradientsG) || gradsHaveNaN(gradientsD)
            warning('[迭代 %d] 梯度含 NaN/Inf，跳过本次更新。', iteration);
            continue;
        end

        netG.State = stateG;

        fmVal = extractdata(gather(lossFM));
        if ~isnan(fmVal)
            lastFMLoss = fmVal;
            nanFMCounter = 0;
        else
            nanFMCounter = nanFMCounter + 1;
            if nanFMCounter >= nanFMMaxSilent
                warning('FM loss 连续 NaN %d 次，可能训练不稳定。', nanFMCounter);
                nanFMCounter = 0;
            end
        end

        [gradientsD, ~] = clipGradients(gradientsD, maxGradNormD);
        [gradientsG, ~] = clipGradients(gradientsG, maxGradNormG);

        [netD, trailingAvgD, trailingAvgSqD] = adamupdate(netD, gradientsD, ...
            trailingAvgD, trailingAvgSqD, iteration, ...
            learnRateD, gradientDecayFactor_D, squaredGradientDecayFactor);
        [netG, trailingAvgG, trailingAvgSqG] = adamupdate(netG, gradientsG, ...
            trailingAvgG, trailingAvgSqG, iteration, ...
            learnRateG, gradientDecayFactor_G, squaredGradientDecayFactor);

        netG_ema = updateEMA(netG_ema, netG, emaBeta);

        % 生成验证图像
        if (mod(iteration, validationFrequency) == 0 || iteration == 1) && ishandle(hFig)
            XGeneratedValidation = predict(netG_ema, ZValidation);
            genVals = extractdata(gather(XGeneratedValidation));
            genMean = mean(genVals(:));
            genStd  = std(genVals(:));
            genStatsHistory.mean(end+1) = genMean;
            genStatsHistory.std(end+1)  = genStd;

            % 崩溃检测
            if iteration > anomalyWarmup ...
                    && length(genStatsHistory.mean) >= genStatsHistory.windowSize + 1
                recentMeans  = genStatsHistory.mean(end - genStatsHistory.windowSize + 1 : end);
                baselineMean = mean(recentMeans);
                meanShift    = abs(genMean - baselineMean);
                stdCollapse  = genStd < anomalyStdMin;

                if meanShift > anomalyMeanThreshold || stdCollapse
                    fprintf('\n!!! [迭代 %d] 检测到生成图像崩溃 !!!\n', iteration);
                    fprintf('   均值偏移: %.4f | 当前 Std: %.4f\n', meanShift, genStd);

                    if crashCount < maxRecoveries
                        checkpointFiles = dir(fullfile(checkpointDir, 'checkpoint_iter_*.mat'));
                        if ~isempty(checkpointFiles)
                            [~, sortIdx] = sort([checkpointFiles.datenum], 'descend');
                            latestCP = fullfile(checkpointDir, checkpointFiles(sortIdx(1)).name);
                            fprintf('   从检查点恢复: %s\n', latestCP);
                            loaded = load(latestCP, ...
                                'netG','netD','netG_ema','iteration','epoch', ...
                                'trailingAvgG','trailingAvgSqG','trailingAvgD','trailingAvgSqD');
                            netG = loaded.netG;  netD = loaded.netD;
                            if isfield(loaded, 'netG_ema'), netG_ema = loaded.netG_ema; end
                            iteration = loaded.iteration;  epoch = loaded.epoch;
                            trailingAvgG  = loaded.trailingAvgG;
                            trailingAvgSqG = loaded.trailingAvgSqG;
                            trailingAvgD  = loaded.trailingAvgD;
                            trailingAvgSqD = loaded.trailingAvgSqD;
                            genStatsHistory.mean = [];  genStatsHistory.std = [];

                            % 重建 mbq
                            mbq = minibatchqueue(augimds, ...
                                'MiniBatchSize', miniBatchSize, ...
                                'PartialMiniBatch', 'discard', ...
                                'MiniBatchFcn', @preprocessMiniBatch, ...
                                'MiniBatchFormat', 'SSCB');
                            shuffle(mbq);

                            % 重建监视器
                            monitor = createMonitor();

                            crashCount = crashCount + 1;
                            fprintf('   恢复完成（%d/%d），mbq + 监视器已重建，从迭代 %d 继续。\n\n', ...
                                crashCount, maxRecoveries, iteration);
                            continue;
                        else
                            fprintf('   警告: 未找到检查点，无法恢复。\n');
                        end
                    else
                        fprintf('   已达最大恢复次数（%d），停止训练。\n', maxRecoveries);
                        monitor.Stop = true;  break;
                    end
                end
            end

            I = imtile(genVals);  I = rescale(I);
            set(hIm, 'CData', I);
            title(hAx, sprintf('Iter %d | Ep %d | lr_G=%.1e | G=%.3f D=%.3f | FM=%.4f', ...
                iteration, epoch, learnRateG, scoreG, scoreD, lastFMLoss));
            drawnow limitrate;
        end

        % 保存检查点
        if mod(iteration, checkpointFrequency) == 0
            checkpointFile = fullfile(checkpointDir, ...
                sprintf('checkpoint_iter_%d.mat', iteration));
            try
                save(checkpointFile, 'netG','netD','netG_ema','iteration','epoch', ...
                    'trailingAvgG','trailingAvgSqG','trailingAvgD','trailingAvgSqD', ...
                    'learnRateD','learnRateG','scoreG','scoreD');
                fprintf('[迭代 %d] 检查点已保存。\n', iteration);

                cpList = dir(fullfile(checkpointDir, 'checkpoint_iter_*.mat'));
                if numel(cpList) > maxCheckpoints
                    [~, cpIdx] = sort([cpList.datenum]);
                    for k = 1:(numel(cpList) - maxCheckpoints)
                        delete(fullfile(checkpointDir, cpList(cpIdx(k)).name));
                    end
                end
            catch ME
                fprintf('[迭代 %d] 检查点保存失败: %s\n', iteration, ME.message);
            end

            % 更新最佳模型
            XValCheck = predict(netG_ema, ZValidation);
            YValCheck = predict(netD, XValCheck);
            valScoreG = extractdata(gather(mean(sigmoid(YValCheck), 'all')));

            if valScoreG > bestScoreG
                bestScoreG    = valScoreG;
                bestIteration = iteration;
                bestFile = fullfile(checkpointDir, 'best_model.mat');
                try
                    save(bestFile, 'netG','netD','netG_ema','iteration','epoch','valScoreG');
                    fprintf('  >> 新最佳模型! Val_G_Score=%.4f (迭代 %d, EMA)\n', ...
                        valScoreG, iteration);
                catch ME
                    fprintf('  最佳模型保存失败: %s\n', ME.message);
                end
            end
        end

        recordMetrics(monitor, iteration, ...
            'GeneratorScore',    scoreG, ...
            'DiscriminatorScore', scoreD, ...
            'LearningRateG',     learnRateG, ...
            'FeatureMatchLoss',  lastFMLoss);
        updateInfo(monitor, 'Epoch', epoch, 'Iteration', iteration);
        monitor.Progress = 100 * iteration / numIterations;
    end
end

fprintf('\n========== 训练结束 ==========\n');
fprintf('总迭代数: %d | 最佳迭代: %d | 最佳 Val_G_Score: %.4f\n', ...
    iteration, bestIteration, bestScoreG);
fprintf('自动崩溃恢复次数: %d\n', crashCount);

%% 11. 保存最终模型
finalModelFile = fullfile(checkpointDir, 'final_model.mat');
save(finalModelFile, 'netG','netD','netG_ema','iteration','epoch');
fprintf('最终模型已保存: %s (Iter=%d, Epoch=%d)\n', finalModelFile, iteration, epoch);

%% 12. 加载最佳模型展示
bestModelFile = fullfile(checkpointDir, 'best_model.mat');
if isfile(bestModelFile)
    loadedBest = load(bestModelFile, 'netG','netG_ema','iteration','valScoreG');
    if isfield(loadedBest, 'netG_ema')
        netG_display = loadedBest.netG_ema;
    else
        netG_display = loadedBest.netG;
    end
    fprintf('加载最佳模型展示: Iter=%d, Val_G_Score=%.4f\n', ...
        loadedBest.iteration, loadedBest.valScoreG);
else
    netG_display = netG_ema;
    fprintf('未找到最佳模型，使用最终 EMA 模型展示。\n');
end

%% 13. 最终生成展示
numObservationsNew = 25;
ZNew = randn(1, 1, numLatentInputs, numObservationsNew, 'single');
ZNew = dlarray(ZNew, 'SSCB');
if canUseGPU, ZNew = gpuArray(ZNew); end
XNew = predict(netG_display, ZNew);
Inew = imtile(extractdata(XNew));  Inew = rescale(Inew);
figure('Name','最终生成结果');  image(Inew);  axis off;
title(sprintf('DCGAN 汉堡生成（转置卷积 kernel=4, λFM=%d, EMA）', lambdaFM));

fprintf('\n训练脚本执行完毕。\n');
fprintf('最终模型: %s\n', finalModelFile);
if isfile(bestModelFile)
    fprintf('最佳模型: %s\n', bestModelFile);
end

%% ==================== 辅助函数 ====================

function mon = createMonitor()
    mon = trainingProgressMonitor( ...
        'Metrics', {'GeneratorScore','DiscriminatorScore','LearningRateG','FeatureMatchLoss'}, ...
        'Info', {'Epoch','Iteration'}, ...
        'XLabel', 'Iteration');
    groupSubPlot(mon, 'Score', {'GeneratorScore','DiscriminatorScore'});
    groupSubPlot(mon, 'Schedule', {'LearningRateG','FeatureMatchLoss'});
end

function [lossG, lossD, gradientsG, gradientsD, stateG, scoreG, scoreD, lossFM] = ...
    modelLoss(netG, netD, X, Z, featLayerName, lambdaFM)

    YReal = forward(netD, X);
    [XGenerated, stateG] = forward(netG, Z);
    YGenerated = forward(netD, XGenerated);

    YRealProb      = sigmoid(YReal);
    YGeneratedProb = sigmoid(YGenerated);
    scoreD = (mean(YRealProb, 'all') + mean(1 - YGeneratedProb, 'all')) / 2;
    scoreG = mean(YGeneratedProb, 'all');

    labelSmooth = 0.9;
    lossD = 0.5 * mean((YReal - labelSmooth).^2, 'all') + ...
            0.5 * mean(YGenerated.^2, 'all');
    lossG_adv = 0.5 * mean((YGenerated - 1).^2, 'all');

    featReal_D      = predict(netD, X,          'Outputs', featLayerName);
    featGenerated_D = predict(netD, XGenerated, 'Outputs', featLayerName);
    lossFM = mean((featReal_D - featGenerated_D).^2, 'all');

    lossG = lossG_adv + lambdaFM * lossFM;

    gradientsG = dlgradient(lossG, netG.Learnables, 'RetainData', true);
    gradientsD = dlgradient(lossD, netD.Learnables);
end

function X = preprocessMiniBatch(data)
    X = cat(4, data{:});
    X = rescale(X, -1, 1, 'InputMin', 0, 'InputMax', 255);
end

function tf = gradsHaveNaN(gradients)
    params = gradients.Value;
    for i = 1:numel(params)
        if any(~isfinite(extractdata(params{i})), 'all')
            tf = true;
            return;
        end
    end
    tf = false;
end

function [gradients, normVal] = clipGradients(gradients, maxNorm)
    normVal = globalGradNorm(gradients);
    if normVal > maxNorm
        scale = maxNorm / normVal;
        gradients = dlupdate(@(g) g .* scale, gradients);
    end
end

function gn = globalGradNorm(gradients)
    gn = 0;
    params = gradients.Value;
    for i = 1:numel(params)
        gn = gn + sum(params{i}.^2, 'all');
    end
    gn = sqrt(double(gather(gn)));
end

function netG_ema = updateEMA(netG_ema, netG, beta)
    params_ema = netG_ema.Learnables.Value;
    params_cur = netG.Learnables.Value;
    for i = 1:numel(params_ema)
        params_ema{i} = beta * params_ema{i} + (1 - beta) * params_cur{i};
    end
    netG_ema.Learnables.Value = params_ema;
end
