clear; close all; 
%% set the parameter 
participant_name='yuwei_hand2_175559';
input_folder=['E:\wait_for_classification' '\' participant_name];
rootFolder4copy='E:\auto_classification'; %new floder
%% set the parameter for a C-scan
start_number=436;
total_number=500;
spacing=4;
%% load folder
rootFolder = [input_folder '\'  participant_name '_plot_500frame']; %'plot' folder
%rootFolder1 = [input_folder '\'  'OAC_log_ham_pgm_resize_500frame'];
%load the folder that you need
OCT_raw_image_512_Folder=[input_folder '\512resize_OCT_log_ham_pgm_resize_500frame'];
OAC_image_512_Folder=[input_folder '\512resize_OAC_log_ham_pgm_resize_500frame'];
OAC_image_1024_Folder=[input_folder '\OAC_log_ham_pgm_resize_500frame'];
Mask_Folder=[input_folder '\' participant_name '_resize512_epidermis_mask_500frame'];
Mask_Folder1=[input_folder '\' participant_name '_epidermis_mask_resize_500frame'];
epidermis_mat_Folder=[input_folder '\' participant_name '_epidermis_mat_500frame'];
% OCT_raw_image_512_Info = dir([OCT_raw_image_512_Folder '\*.pgm']); %need to sort?
% OAC_image_512_Info = dir([OAC_image_512_Folder '\*.pgm']);
OAC_image_1024_Info = dir([OAC_image_1024_Folder '\*.pgm']);
 
% Mask_Info = dir([Mask_Folder '\*.png']);
% Mask_T = struct2table(Mask_Info);
% Mask_sortedT = sortrows(Mask_T,'date'); 
% Mask_sortedS = table2struct(Mask_sortedT);

% epidermis_mat_Info = dir([epidermis_mat_Folder '\*.mat']);
% epidermis_mat_T = struct2table(epidermis_mat_Info);
% epidermis_mat_sortedT = sortrows(epidermis_mat_T,'date'); 
% epidermis_mat_sortedS = table2struct(epidermis_mat_sortedT);

%% creat new floder for the calssfication
rootFolder4copy_correct = [rootFolder4copy '\'  participant_name '\correct']; 
rootFolder4copy_modification = [rootFolder4copy '\' participant_name '\wrong']; 
OCT_image_size512_correct=[rootFolder4copy_correct '\OCT_image_size512'];
OCT_image_size512_modification=[rootFolder4copy_modification '\OCT_image_size512'];
OAC_image_size512_correct=[rootFolder4copy_correct '\OAC_image_size512'];
OAC_image_size512_modification=[rootFolder4copy_modification '\OAC_image_size512'];
%OAC_image_size1024_correct=[rootFolder4copy_correct '\OAC_image_size1024' ];
OAC_image_size1024_modification=[rootFolder4copy_modification '\OAC_image_size1024'];
mask_for_correct=[rootFolder4copy_correct '\Mask512'];
mask_for_correct1=[rootFolder4copy_correct '\Mask'];
epidermis_mat_for_wrong=[rootFolder4copy_modification '\epidermis_mat'];
mask512_for_wrong=[rootFolder4copy_modification '\mask512'];
mask_for_wrong=[rootFolder4copy_modification '\mask'];
%%
imgsInfo = dir([rootFolder '\*.png']);
%imgsInfo1 = dir([rootFolder1 '\*.pgm']);
T = struct2table(imgsInfo);
sortedT = sortrows(T,'date'); 
sortedS = table2struct(sortedT);

figure(1),title('read to load images and review', 'fontsize', 14);
% set(gcf,'Position',[100 100 500 500])
f = figure(1);
f.Position = [400 125 1200 750];

flag4copy1 = 0;
flag4copy2 = 0;
count_for_wrong_image=0; 
count_for_correct_image=0;

%% excel default 
summation_excel=cell(total_number/spacing,5);
summation_excel{1,1}='correct image';
summation_excel{2,1}='Numerical order';
summation_excel{1,2}='wrong image';
summation_excel{2,2}='Numerical order';
summation_excel{1,3}='total number of correct image';
summation_excel{1,4}='total number of wrong image';
summation_excel{1,5}='Accuracy';
%%
for i = start_number:spacing:total_number
    j = i/4;
    repeat_button=1;
    while (repeat_button==1)
        %i = 1:size(sortedS, 1)
        imgName = [rootFolder '\' num2str(j) '.png'];
%         imgName1 = [rootFolder1 '\' participant_name '_' num2str(i,'%04d') '.pgm'];
        img = imread(imgName);
        %img2 = img(150:600,:,:);
        figure(1);
        imagesc(img),hold on; 
        title(['img #:' num2str(i) ' correct->space;  wrong -> left arrow' newline 'show OAC image -> right arrow'], 'fontsize', 14);
        %callback function
        OCT_512imgName = [OCT_raw_image_512_Folder '\' participant_name '_' num2str(i,'%04d') '.pgm'];
        OCT_512imgName2=imread(OCT_512imgName);
        OAC_512imgName = [OAC_image_512_Folder '\'  participant_name '_' num2str(i,'%04d') '.pgm'];
        OAC_512imgName2=imread(OAC_512imgName);
        OAC_1024imgName = [OAC_image_1024_Folder '\'  participant_name '_' num2str(i,'%04d') '.pgm'];
        OAC_1024imgName2=imread(OAC_1024imgName);
        %Mask_imgName = [Mask_Folder '\' Mask_sortedS(i).name];
        Mask_imgName = [Mask_Folder '\' participant_name '_' num2str(j) '.png'];
        Mask_imgName2=imread(Mask_imgName);
        Mask1_imgName = [Mask_Folder1 '\' participant_name '_' num2str(j) '.png'];
        Mask1_imgName2=imread(Mask1_imgName);
        %epidermis_mat=[epidermis_mat_Folder '\' epidermis_mat_sortedS(i).name];
        epidermis_mat=[epidermis_mat_Folder '\' participant_name '_epidermis_' num2str(j) '.mat'];
        load(epidermis_mat);
        epidermis_mat2=epidermis;
        pause;
    %isKeyPressed = ~isempty(get(f,'CurrentCharacter'));
    %key_press=waitforbuttonpress;
        isKeyPressed = double(get(f,'CurrentCharacter'));
    %set(f, 'KeyPressFcn', @(x,y)disp(get(f,'CurrentCharacter')));
        if strcmp(num2str(isKeyPressed),'28')==1 %for wrong image
             disp('wrong image');
             if ~flag4copy1
                mkdir(OCT_image_size512_modification);
                mkdir(OAC_image_size512_modification);
                mkdir(OAC_image_size1024_modification);       
                mkdir(epidermis_mat_for_wrong);
                mkdir(mask512_for_wrong);
                mkdir(mask_for_wrong)
             end
             %OAC_image_512 = dir([OAC_image_512_Folder '\*' i '.pgm']);
             imgName4copy4 = [OCT_image_size512_modification '\' participant_name '_' num2str(i,'%04d') '.pgm']; %sortedS(i).name
             imwrite(OCT_512imgName2,imgName4copy4);
             imgName4copy = [OAC_image_size512_modification '\' participant_name '_' num2str(i,'%04d') '.pgm'];
             imwrite(OAC_512imgName2, imgName4copy); 
             imgName4copy2 = [OAC_image_size1024_modification '\' participant_name '_' num2str(i,'%04d') '.pgm'];
             imwrite(OAC_1024imgName2, imgName4copy2); 
             imgName4copy3 = [epidermis_mat_for_wrong '\' participant_name '_' num2str(j,'%04d') '.mat'];
             save(imgName4copy3,'epidermis_mat2'); 
             imgName4copy5 = [mask512_for_wrong '\' participant_name '_' num2str(j,'%04d') '.png'];
             imwrite(Mask_imgName2, imgName4copy5);
             imgName4copy6 = [mask_for_wrong '\' participant_name '_' num2str(j,'%04d') '.png'];
             imwrite(Mask1_imgName2, imgName4copy6);
             flag4copy1 = 1;
             count_for_wrong_image=count_for_wrong_image+1;
             summation_excel{count_for_wrong_image+2,2}=num2str(i);
             repeat_button=0;
             
        elseif strcmp(num2str(isKeyPressed),'32')==1 %for correct image
               disp('correct image');
               if ~flag4copy2
                   mkdir(OCT_image_size512_correct);
                   mkdir(OAC_image_size512_correct);
                   %mkdir(OAC_image_size1024_correct);
                   mkdir(mask_for_correct);
                   mkdir(mask_for_correct1);
               end
               imgName4copy_4 = [OCT_image_size512_correct '\' participant_name '_' num2str(i,'%04d') '.pgm'];
               imwrite(OCT_512imgName2, imgName4copy_4);
               imgName4copy_1 = [OAC_image_size512_correct '\' participant_name '_' num2str(i,'%04d') '.pgm'];
               imwrite(OAC_512imgName2, imgName4copy_1);     %pgm and jpg   
               %imgName4copy_2 = [OAC_image_size1024_correct '\' participant_name '_' num2str(i) '.pgm'];
               %imwrite(OAC_1024imgName2, imgName4copy_2);
               imgName4copy_3 = [mask_for_correct '\' participant_name '_' num2str(j,'%04d') '.png'];
               imwrite(Mask_imgName2, imgName4copy_3);
               imgName4copy_5 = [mask_for_correct1 '\' participant_name '_' num2str(j,'%04d') '.png'];
               imwrite(Mask1_imgName2, imgName4copy_5);
               flag4copy2 = 1;         
               count_for_correct_image=count_for_correct_image+1;
               summation_excel{count_for_correct_image+2,1}=num2str(i); 
               repeat_button=0;
        elseif strcmp(num2str(isKeyPressed),'29')==1  %show image without line
               figure(10)
               OAC_1024imgName2=imread(OAC_1024imgName);
               imagesc(OAC_1024imgName2),colormap(gray);
               repeat_button=1;
               f1=figure(10);
               f1.Position = [750 250 1200 350];
               pause;
        elseif (isKeyPressed~=28 && isKeyPressed~=32 && isKeyPressed~=29)
               repeat_button=1;
        end
    end
  
end

h = figure(1);
close(h);
%% save summation excel
summation_excel{2,3}=count_for_correct_image;
summation_excel{2,4}=count_for_wrong_image;
summation_excel{2,5}=count_for_correct_image/(count_for_wrong_image+count_for_correct_image);
summation_filename= [rootFolder4copy '\' participant_name '\' participant_name '.xls'];
xlswrite(summation_filename,summation_excel);
%% bar chart 
bar_y=zeros(1,total_number/spacing);
for ii=3:1:count_for_correct_image+2
    bar_y(1,(str2double(summation_excel{ii,1}))/spacing)=1;
end
figure(2);
bar_image=bar(bar_y);
title('distribution of correct image');
xlabel('total number: 250 images'); %real number = number*2
ylabel('correct label');
ImName_bar = [rootFolder4copy '\' participant_name '\' participant_name '_bar_chart.fig'] ;
saveas(bar_image,ImName_bar,'fig');

%%
%  function myfun(~,event)
%     disp(event.Key);
%     Keypressed = event.Key;
%     if strcmp(Keypressed,'a') == 1
%          if ~flag4copy
%             mkdir(rootFolder4copy_correct);
%         end
%         imgName4copy = [rootFolder4copy_correct '\' sortedS(i).name];
%         imwrite(img, imgName4copy);     %pgm and jpg    
%         flag4copy = 1;
%         copyCounter = copyCounter + 1;   
%      
%     end
%     
% %     switch key
% %         case 'leftarrow'
% %         
% %         case 'rightarrow'
% %         if ~flag4copy
% %               mkdir(rootFolder4copy_correct);
% %         end
% %         imgName4copy = [rootFolder4copy_correct '\' sortedS(i).name];
% %         imwrite(img, imgName4copy);     %pgm and jpg    
% %         flag4copy = 1;
% %         copyCounter = copyCounter + 1;    
% %     end
%     
%  end
    
 %     [x, y, button] = ginput(1); #bottom
%     if button == 1
% %         disp('image ok');
%     elseif button == 3
% %         disp('copy image');
%         
%         if ~flag4copy
% %         if ~isfile(rootFolder4copy)
%               mkdir(rootFolder4copy_correct);
%         end
%         imgName4copy = [rootFolder4copy_correct '\' sortedS(i).name];
%         imwrite(img, imgName4copy);     %pgm and jpg    
%         flag4copy = 1;
%         copyCounter = copyCounter + 1;
%     end     
