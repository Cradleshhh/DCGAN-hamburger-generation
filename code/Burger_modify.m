% ===== 请修改这两个路径 =====
inputFolder = 'D:\hamberger_datasets\Fast Food Classification V2\Train\Burger'; 
outputFolder = 'D:\hamberger_datasets\Fast Food Classification V2\Train\Burger_modify'; 
outputSize = 256; % 目标尺寸
% =============================

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% 同时搜索 .jpg 和 .jpeg 文件（不区分大小写）
files = [dir(fullfile(inputFolder, '*.jpg')); 
         dir(fullfile(inputFolder, '*.jpeg'))];

fprintf('找到 %d 个图片文件\n', length(files));

for i = 1:length(files)
    img = imread(fullfile(inputFolder, files(i).name));
    [h, w, c] = size(img);
    
    % 1. 计算缩放比例（取较小边，确保完全放入256x256框内）
    scale = min(outputSize / h, outputSize / w);
    newH = round(h * scale);
    newW = round(w * scale);
    resizedImg = imresize(img, [newH, newW]);
    
    % 2. 创建256x256的纯白画布
    if c == 1 % 灰度图
        squareImg = 255 * ones(outputSize, outputSize, 'uint8');
    else % 彩色图
        squareImg = 255 * ones(outputSize, outputSize, c, 'uint8');
    end
    
    % 3. 居中放置（计算偏移量）
    yOffset = floor((outputSize - newH) / 2) + 1;
    xOffset = floor((outputSize - newW) / 2) + 1;
    squareImg(yOffset:yOffset+newH-1, xOffset:xOffset+newW-1, :) = resizedImg;
    
    % 4. 保存
    imwrite(squareImg, fullfile(outputFolder, files(i).name));
    fprintf('已处理：%s\n', files(i).name);
end

disp('所有图片处理完毕！');