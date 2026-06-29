%% 加载训练好的 DCGAN 模型并生成 25 张汉堡图像
clear; clc;

% 加载模型
load('hamburger_gan_final.mat');

% 推断潜在向量维度
if ~exist('numLatentInputs', 'var')
    numLatentInputs = netG.Layers(1).InputSize(3);
end

% 生成随机噪声
numImages = 25;
ZNew = randn(1, 1, numLatentInputs, numImages, 'single');
ZNew = dlarray(ZNew, 'SSCB');
if canUseGPU
    ZNew = gpuArray(ZNew);
end

% 前向生成
XNew = predict(netG, ZNew);

% 拼贴展示
Inew = imtile(extractdata(XNew));
Inew = rescale(Inew);
figure('Name', '生成的新汉堡');
image(Inew); axis off; title(sprintf('DCGAN 生成汉堡（%d 张）', numImages));
drawnow;

%% 保存图像
answer = questdlg('是否保存生成的图像？', '保存图像', ...
    '是，选择保存路径', '否', '是，选择保存路径');

switch answer
    case '是，选择保存路径'
        saveDir = uigetdir(pwd, '选择保存文件夹');
        if saveDir == 0
            warning('未选择文件夹，图像未保存。');
        else
            timestamp = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
            saveSubDir = fullfile(saveDir, ['generated_burgers_', timestamp]);
            mkdir(saveSubDir);

            genImgs = extractdata(gather(XNew));
            for i = 1:numImages
                if ndims(genImgs) == 4
                    img_i = genImgs(:, :, :, i);
                else
                    img_i = genImgs(:, :, i);
                end
                img_i = rescale(img_i);
                img_i_uint8 = im2uint8(img_i);
                fileName = fullfile(saveSubDir, sprintf('burger_%02d.png', i));
                imwrite(img_i_uint8, fileName);
            end

            montageFile = fullfile(saveSubDir, 'montage_5x5.png');
            imwrite(im2uint8(Inew), montageFile);

            fprintf('已保存 %d 张单图 + 1 张拼贴图到:\n   %s\n', numImages, saveSubDir);
        end

    case '否'
        disp('图像未保存。');
end
