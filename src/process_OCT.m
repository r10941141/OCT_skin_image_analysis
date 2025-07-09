clc;close all;clear;
elapsed_time_images=0;
elapsed_time_mask=0;
fclose('all');
%% detect line
a_scan=1024;
page=125; %need to adjust
detect_junction=1;
%OAC_image=zeros(1024,1000,500); % save oac image %pixels./2 %lines
pixel_size_um=5.92;
fitting_order=3; %polynomial curve fitting
%% 20210626 OAC parameters
UseGaussian = 0; %Gaussian
sigma=1.5; %Gaussian parameter
OAC = 1;
cut = 1;
pixel_size=5.92*10^(-6);%um convert to m %for OAC
Refractive_index_epidermis=1.424;
N=4095; %for zero pandding 8192
%% processing / saving setting
AscanStartInd = 1;  % for the use of removing bad points
flag4hamWin = 1;    % 1 for hamwidow; 0 for no spectral shapping
flag4savingCXOCTimage = 0;  % 1 for saving log scale Bscan image; 0 for not saving
flag4kcalibration = 1 ;%1 for activing the k calibration processing
save_OCT_data = 1; % 1 for saving log scale Bscan image original OCT data; 0 for not saving
save_OAC_data = 1; % 1 for saving log scale Bscan image original OAC data; 0 for not saving
flag4savingLinearVolas3Dtiff = 1;
% flag2saveComplex = 0;
flag4ZeroPadding = 1;
DyRange = 30; % setting dynamic range of the log scale OCT image
MaxImg = 5e5;
MinImg = 0;
flag4linear2save = 1;
%% automatic batch processing
% participant='Shi_nose';
background_line = 0; %need to adjust %background setting
hostfolder ='..\data\public\raw_data'; %raw data path
calibrationFilePath = '..\data\public\k_calibration_new.mat'; 

use_external_mask = 1; 
mask_source_type = 'model'; % 'model' or 'GT'

maskpath = '..\data\mask_path'; 

if flag4kcalibration == 1
   load(calibrationFilePath);
end
% binBase = dir([hostfolder '\*.bin']);
binBase = dir([hostfolder '\*info.txt']);
h = waitbar(0, 'processing initiated'); % creat processing status indicator

for binBaseInd = 1:length(binBase(:))
    %for binBaseInd = 1:1
    u_z3_cell=cell(page,1);
    OAC_image_cell=cell(page,1);
    TH=zeros(801,page);
    Date=hostfolder(end-7:end);
    fname = binBase(binBaseInd).name;
    underscore_inds = strfind(fname, '_');
    sampleNoInd = strfind(fname, '_S_');
    ascanNoInd = strfind(fname, '_L_');
    bscanNoInd = strfind(fname, '_F_');
    endNoInd = strfind(fname, '_Re_');
    binIDIndexA=find(underscore_inds==(strfind(fname, Date)+8));
    binID = fname(underscore_inds(binIDIndexA)+1:underscore_inds(binIDIndexA+1)-1);
    pixels = str2double(fname(sampleNoInd+3:ascanNoInd-1));
    lines = str2double(fname(ascanNoInd+3:bscanNoInd-1));
    framesTotal = str2double(fname(bscanNoInd+3:endNoInd-1));
    
    %% preparing file loading /image saving
    %Hamwin = hamming(pixels-AscanStartInd+1);   % hamming window for spectral shapping
    Hamwin=hamming(pixels);
    Hamwin=single(Hamwin)';
    if flag4ZeroPadding ==1
        zeropadnum = 8192;  % the FFT size
    else
        zeropadnum=pixels;
    end
%     if flag2saveComplex==1
%         %img = complex(zeros(zeropadnum/2, lines, framesTotal, 'double'));
%         img = complex(zeros( zeropadnum/2, lines, framesTotal, 'double'));
%     else
%         %img = zeros(zeropadnum/2, lines, framesTotal, 'double');
%         img = zeros( zeropadnum/2, lines, framesTotal, 'double');
%     end
%     zeropixel = zeros(1, int16((zeropadnum - pixels)/2));   % prepared for dispersion compensation use if
    if binIDIndexA==1
        FoderName=binID;
    else
        FoderName=[fname(1:underscore_inds(binIDIndexA-1)),binID];
    end
    if flag4hamWin==1
        Folder2savelog = [hostfolder '\' FoderName '\' mask_source_type 'OAC_log_ham_pgm_resize_500frame'];
        Folder2savelog2 = [hostfolder '\' FoderName '\' mask_source_type '512resize_OAC_log_ham_pgm_resize_500frame'];
        Folder2savelog3= [hostfolder '\' FoderName '\OCT_log_ham_pgm_resize_500frame'];
        Folder2savelog4 = [hostfolder '\' FoderName '\512resize_OCT_log_ham_pgm_resize_500frame'];
%         Folder2savelinear = [hostfolder '\' FoderName '\linear_ham_tiff'];

    else
        Folder2savelog = [hostfolder '\' FoderName '\log_pgm'];
%         Folder2savelinear = [hostfolder '\' FoderName '\linear_tiff'];

    end
    
   


    if ~exist(Folder2savelog)
        mkdir(Folder2savelog);
    end 
    if ~exist(Folder2savelog2)
        mkdir(Folder2savelog2);
    end 
     if ~exist(Folder2savelog3)
        mkdir(Folder2savelog3);
     end 
      if ~exist(Folder2savelog4)
        mkdir(Folder2savelog4);
    end 
%     if ~exist(Folder2savelinear)
%         mkdir(Folder2savelinear);
%     end

    %% per bin file
    framenumCount = 0;
    subplotframe=0; 
    subBinBase = dir([hostfolder '\' fname(1:underscore_inds(binIDIndexA+1)) '*.bin']);
    
    for subBinBaseInd = 1:length(subBinBase(:))
        %    for subBinBaseInd = 1:1
        framenumCountStart=framenumCount+1;
        TmpSubBinBase = dir([hostfolder '\' fname(1:underscore_inds(binIDIndexA+1)) '*_' num2str(subBinBaseInd-1) '.bin']);
        subFname = TmpSubBinBase(1).name;
        
        sampleNoInd = strfind(subFname, '_S_');
        ascanNoInd = strfind(subFname, '_L_');
        bscanNoInd = strfind(subFname, '_F_');
        endNoInd = strfind(subFname, '_Re_');
        
        pixels = str2double(subFname(sampleNoInd+3:ascanNoInd-1));
        lines = str2double(subFname(ascanNoInd+3:bscanNoInd-1));
        frames = str2double(subFname(bscanNoInd+3:endNoInd-1));
        numTestFrames = frames;
%         numTestFrames = 6;
        
        %% loading individual bin files for each vol data
        waitbar((subBinBaseInd-1)/length(subBinBase), h, ['loading ' num2str( binID) ' bin #: ' num2str(subBinBaseInd) '/' num2str(length(numTestFrames))]);
%         disp(['loading ' num2str( binID) ' bin #: ' num2str(subBinBaseInd) '/' num2str(length(numTestFrames))])
        filename = [hostfolder '\' subFname];
        fid = fopen(filename, 'r');
        a = single(fread(fid, [pixels lines*numTestFrames], 'uint16'));
        %         a = single(fread(fid, [pixels lines*frames], 'uint16'));
        a = reshape(a, [pixels lines numTestFrames]);
        % added 2018-08-01 remove bad points
        a = a(AscanStartInd:end, :, :);

 
        if subBinBaseInd == 1
            %     BG = mean(squeeze(a(:, 1:100, 1)), 2);  % average the first 100 lines as the background
            BG = (mean(squeeze(a(:, 1, 1:numTestFrames)), 2));
        end
        %         filenameBG = 'F:\Sherry\20210201\BG\raw_092937\log_ham_pgm.tif';
        %         fidBG = fopen(filenameBG,'r');
        %         g = fread(fidBG, [2048 1000], 'uint8');
        %         BG = mean(g(:, 1:400), 2);
        %         a = a - BG; % background subtraction
        
        fclose('all');
        
        %% dispersion compensation paraemeters
        compens_param1 = 0.0;
        compens_param2 = 0.0;
        compensate1 = single(exp(sqrt(-1)*(compens_param1/1e6)*(linspace(-pixels/2, pixels/2, pixels)).^2));
        compensate = compensate1.*single(exp(sqrt(-1)*(compens_param2/1e9)*(linspace(-pixels/2, pixels/2, pixels)).^3));
        % img = single(zeros(zeropadnum/2,ascanlist(1),frames));

        %% Processing
        % framenum = 1; % re-set the variable back to its initial value
    
        for framenum = 4:4:numTestFrames
            tic;
            %     disp(['frame#: ' num2str(framenum)]);
            tempimg = single(zeros(zeropadnum, lines)); %zeropadnum=8196
            tempa = squeeze(a(:, :,framenum));
            tempa2 = single(zeros(pixels,1000));
            if UseGaussian == 1
               [M,N]=size(tempa);
                gfilter=zeros(M,N);
                for i=1:M
                  for j=1:N
                  dist= (i-M/2)^2 + (j-N/2)^2;
                  gfilter(i,j) = 255*exp((-dist)/(2*(sigma)^2));
                  end
                end
    
            %tempa2 = tempa.*gfilter;
                tempa2 =conv2(tempa,gfilter,'same');
                tempa2=tempa2./(2000);
            else
                tempa2(:,:)=tempa(:,:);
            end
            clear tempa;
            for linenum = 1:lines
                k_fringe = (tempa2(:, linenum) - BG)';
                if flag4kcalibration == 1
                    k_fringe = interp1(single(0:1/(pixels-1):1), k_fringe, single(k_calibration)', 'spline');    
                end
                if flag4hamWin == 1
                    k_fringe=k_fringe.*Hamwin.*compensate;
                    if flag4ZeroPadding==1
                        k_fringe2=[zeros(1,ceil((zeropadnum-pixels)/2)) k_fringe zeros(1,floor((zeropadnum-pixels)/2))];
                        tempimg(:, linenum) = fft(k_fringe2);    % 2018-08-06 spectral shapping added HCL
                    else
                        tempimg(:, linenum) = fft(k_fringe); 
                    end
                   
                else
                    k_fringe=k_fringe.*compensate;
                    if flag4ZeroPadding==1
                        k_fringe2=[zeros(1,ceil((zeropadnum-pixels)/2)) k_fringe zeros(1,floor((zeropadnum-pixels)/2))];
                        tempimg(:, linenum) = fft(k_fringe2);    % 2018-08-06 spectral shapping added HCL
                    else
                        tempimg(:, linenum) = fft(k_fringe); 
                    end
                end
                %         tempimg(:,linenum) = fft(complex(k_fringe).*compensate.*GaussCorr, zeropadnum);
                %         tempimg(:,linenum) = fft(complex(k_fringe).*GaussCorr, zeropadnum);
            end
            denoise = min(min(tempimg(end-50:end,2:end)));
            tempimg(:,:)=tempimg(:,:)-denoise;
            
            if OAC == 1  
               % detect line from mask 
               mask_num = framenum/4;
               mask_num1 = num2str(mask_num ,'%04d');

               mask_path = [maskpath FoderName '_' mask_num1 '.png'];
               mask_data = imread(mask_path);
               T1=zeros(1,944);
              
               for x=1:1:944
                   has_data = mask_data(:,x) > 127;
                   first_data_pos = find(has_data, 1, 'first');
                   T1(1,x)=first_data_pos;
               end
                

               ratio=zeros(1,1000);
               tempimg2=zeros(4096,lines);
               tempimg3=zeros(4096,lines);
               u_z2=zeros(4096,lines);tempimg;
               %intensity2=zeros(4096,lines);
               %intensity3=zeros(4096,lines);  
               tempimg=abs(tempimg); 
               fval=zeros(4096,1);
               tempimg2(:,:)=tempimg(1:4096,:);
               tempimg3(:,:)=tempimg(1:4096,:);
               %find gradient and find the surface line
               T2=squeeze(10.*log10(tempimg3(1:2000,:)));
               T2=uint16(round(T2-mean(T2(end/2:end,:),'all')).*65536./DyRange);
               [thr,sorh,keepapp] = ddencmp('den','wv',T2);
               ixc=wdencmp('gbl',T2,'sym4',6,thr,sorh,keepapp);
               k2=medfilt2(ixc,[15 15]);
               isuo=imresize(k2,1, 'bicubic' );
               [Gxx, Gyy]=imgradientxy(isuo,'sobel');
               
               if background_line>0
                  Gyy((background_line*4)-10:(background_line*4)+10,:)=0;
               end
               %     figure(1),subplot(2,2,1),imagesc(ixc),title('denoise '),hold on;
               %     subplot(2,2,2),imagesc(isuo);title( 'medium filter' ),hold on;
               %     subplot(2,2,4),imagesc(T2);title('T2'),hold on;
               X_surface=zeros(1,lines);
               for x=1:1:lines 
                    [M_1,N_1]=max(Gyy(20:end,x)); %Gy(1:130,x)
                    X_surface(1,x)=N_1+20; 
               end     
               subplotframe= subplotframe+1;
               for x=6:1:lines-10
                   if abs(X_surface(:,x-1)-X_surface(:,x))>20
                       X_surface(:,x)=(X_surface(:,x-3)+X_surface(:,x+3))/2;
                   elseif abs(X_surface(:,x-3)-X_surface(:,x))>30       
                          X_surface(:,x)=(X_surface(:,x-5)+X_surface(:,x+5))/2;
                   else
                        X_surface(:,x)=X_surface(:,x);
                   end
               end

               if use_external_mask == 1
                   mask_num = framenum/4;
                   mask_num1 = num2str(mask_num ,'%04d');
    
                   mask_path = [maskpath FoderName '_' mask_num1 '.png'];
                   mask_data = imread(mask_path);
                   T1=zeros(1,944);
                  
                   for x=1:1:944
                       has_data = mask_data(:,x) > 127;
                       first_data_pos = find(has_data, 1, 'first');
                       T1(1,x)=first_data_pos;
                       
                   X_surface(1,30:end-27) = T1*4 + 40;    
                   end
               end

               
%                plot(X(:,:)),set(gca,'YDir','reverse');
%                ready to process OAC 
               for linee=2:1:lines-1
                   tempimg2(:,linee)=((tempimg(1:4096,linee-1)+tempimg(1:4096,linee)+tempimg(1:4096,linee+1))./3);
                   tempimg2(:,linee)=smooth(tempimg2(:,linee));%10
%                    [M,J]=max(Gy(20:end,linee));
                   J=round(X_surface(1,linee));
                   for z=J:1:J+600
                       fval(z)=sum(tempimg2(z+1:end,linee));    
                       u_z=tempimg2(z,linee)/(2*pixel_size/4/Refractive_index_epidermis*fval(z));
                       u_z2(z,linee)=u_z;
                   end
%                    ratio(1,linee)=max(tempimg2(J:J+600,linee)./u_z2(J:J+600,linee));
               end
               u_z3_cell{subplotframe,1}=u_z2(:,:);
%                for linee=2:1:lines-1 %u is a constant per b-scan
%                    J=round(X_surface(1,linee));
%                    for z=J:1:J+600
% %                        tempimg3(J:J+600,linee)=u_z2(J:J+600,linee).*mean(ratio(1,:));
%                        tempimg3(J:J+600,linee)=u_z2(J:J+600,linee).*10;
%                    end
%                end
               u_z2_ratio=u_z2.*1.76;
               tempimg3(u_z2>0)=double(u_z2_ratio(u_z2_ratio>0));
                
               
%                figure(2)
%                subplot(1,3, subplotframe)
%                plot(normalize(u_z2(200:1000,200)),'r'),hold on,plot(normalize(tempimg2(200:1000,200)),'b'),legend('u(z)','rawdata'),title(['normalization frame ', num2str(framenum)]);
               
%             else
%                 tempimg2=tempimg(1:4096,:);
            end    
           
            ZimageOAC= abs(squeeze(10*log10(tempimg3(1:4096, :)))); %OAC
            noise = mean(ZimageOAC(end/2:end,:),'all'); 
            ZimageRescalOAC = round((ZimageOAC-noise).*65536/DyRange);  % assume 55dB dynamic range
            ZimageRescalFlipOAC = ZimageRescalOAC;

            ZimageOCT= abs(squeeze(10*log10(tempimg(1:4096, :)))); %OAC
            noise = mean(ZimageOCT(end/2:end,:),'all'); 
            ZimageRescalOCT = round((ZimageOCT-noise).*65536/DyRange);  % assume 55dB dynamic range
            ZimageRescalFlipOCT = ZimageRescalOCT;
            % downsize to 1024
            if flag4ZeroPadding == 1
%                multiple=(zeropadnum/pixels);
%                ZimageDownsizeOAC=zeros(pixels./2,lines);
%                ZimageDownsizeOCT=zeros(pixels./2,lines);
%                for depth=1:1:4096
%                    for ascan=1:1:lines
%                        if mod(depth,multiple)==0
%                           ZimageDownsizeOAC(depth./multiple,ascan)=ZimageRescalFlipOAC(depth,ascan);
%                           ZimageDownsizeOCT(depth./multiple,ascan)=ZimageRescalFlipOCT(depth,ascan);
%                        end
%                    end
%                end 
                ZimageDownsizeOAC=ZimageRescalFlipOAC(4:4:end,:);
                ZimageDownsizeOCT=ZimageRescalFlipOCT(4:4:end,:);
            end
            
            a1=real(uint16(ZimageDownsizeOAC));
%             a=real(uint16(ZimageRescalFlip));
            %ImName = [ Folder2savelog '\' FoderName '_' num2str(framenum) '.pgm'];
            OAC_image_cell{subplotframe,1}=a1(:,:);
            a1=a1(10:379,30:end-27);  % if sample is close to DC line, the boudary is neccessory to be changed.
            a2=imresize(a1,[512 512]);

            a3=real(uint16(ZimageDownsizeOCT));
            a3=a3(10:379,30:end-27);  % if sample is close to DC line, the boudary is neccessory to be changed.
            a4=imresize(a3,[512 512]);
            elapsed_time = toc;
            elapsed_time_images = elapsed_time_images + elapsed_time;
            disp(['總執行時間(image)：', num2str(elapsed_time_images), ' 秒']);
%             ImName_image = [hostfolder '\' FoderName 'OAC_image' '.mat'] ;
%             save(ImName_image,'OAC_image');
             clear ZimageDownsize; 
             clear ZimageRescalFlip;
             clear ZimageRescal;
             clear Zimage; 

             clear tempimg3;
% %             clear tempimg2;
             clear tempa2;
             clear tempa;
% %             clear u_z2;
%             clear intensity2;
%             clear intensity3;
%             clear Gxx Gyy k2;
            %a2=imresize(a1,[128 128]);
            %imwrite(a1,ImName,'pgm');
            if framenum>0
                %         ImName = [Folder2save '\' FileTag '_' num2str(framenum-4, '%4.4d\n') '_'  num2str(compens_param2+200, '%4.4d\n') '.pgm'];
                %         ImName = [Folder2save '\' FileTag '_' num2str(framenum-4, '%4.4d\n') '.pgm'];
                %         ImName = [ Folder2save '\' num2str(framenum, '%4.4d\n') '.tif'];
                if flag4hamWin == 1  && subBinBaseInd == 1
                    ImName = [ Folder2savelog '\' FoderName '_' num2str(framenum, '%4.4d\n') '.pgm'];
                    ImName2 = [ Folder2savelog2 '\' FoderName '_' num2str(framenum, '%4.4d\n') '.pgm'];
                    ImName3 = [ Folder2savelog3 '\' FoderName '_' num2str(framenum, '%4.4d\n') '.pgm'];
                    ImName4 = [ Folder2savelog4 '\' FoderName '_' num2str(framenum, '%4.4d\n') '.pgm'];
                elseif flag4hamWin == 1  && subBinBaseInd ~= 1
                    ImName = [ Folder2savelog '\' FoderName '_' num2str(framenum + framenumCount, '%4.4d\n') '.pgm'];
                    ImName2 = [ Folder2savelog2 '\' FoderName '_' num2str(framenum + framenumCount, '%4.4d\n') '.pgm'];
                    ImName3 = [ Folder2savelog3 '\' FoderName '_' num2str(framenum + framenumCount, '%4.4d\n') '.pgm'];
                    ImName4 = [ Folder2savelog4 '\' FoderName '_' num2str(framenum + framenumCount, '%4.4d\n') '.pgm'];
                elseif subBinBaseInd == 1
                    ImName = [ Folder2savelog '\' FoderName '_'  num2str(framenum, '%4.4d\n') '.pgm'];
                    ImName2 = [ Folder2savelog2 '\' FoderName '_'  num2str(framenum, '%4.4d\n') '.pgm'];
                    ImName3 = [ Folder2savelog3 '\' FoderName '_'  num2str(framenum, '%4.4d\n') '.pgm'];
                    ImName4 = [ Folder2savelog4 '\' FoderName '_'  num2str(framenum, '%4.4d\n') '.pgm'];
                elseif subBinBaseInd ~=1
                    ImName = [ Folder2savelog '\' FoderName '_'  num2str(framenum + framenumCount, '%4.4d\n') '.pgm'];
                    ImName2 = [ Folder2savelog2 '\' FoderName '_'  num2str(framenum + framenumCount, '%4.4d\n') '.pgm'];
                    ImName3 = [ Folder2savelog3 '\' FoderName '_'  num2str(framenum + framenumCount, '%4.4d\n') '.pgm'];
                    ImName4 = [ Folder2savelog4 '\' FoderName '_'  num2str(framenum + framenumCount, '%4.4d\n') '.pgm'];
                end
                if flag4savingCXOCTimage == 1
                   %imwrite(real(uint16(ZimageDownsize(:,:))), ImName, 'pgm'); %0701
                   imwrite(real(uint16(a1(:,:))), ImName, 'pgm'); %0701
                   imwrite(real(uint16(a2(:,:))), ImName2, 'pgm'); %0701
                   imwrite(real(uint16(a3(:,:))), ImName3, 'pgm'); %0701
                   imwrite(real(uint16(a4(:,:))), ImName4, 'pgm'); %0701
                end
            end
            if subBinBaseInd == 1
                waitbar(framenum/framesTotal, h, ['processing ' num2str(binID) ' frame #: ' num2str(framenum) '/' num2str(framesTotal)]);
%                 disp(['processing ' num2str(binID) ' frame #: ' num2str(framenum) '/' num2str(framesTotal)])
            else
                waitbar((framenum+framenumCount)/framesTotal, h, ['processing ' num2str(binID) ' frame #: ' num2str(framenum + framenumCount) '/' num2str(framesTotal)]);
%                 disp(['processing ' num2str(binID) ' frame #: ' num2str(framenum + framenumCount) '/' num2str(framesTotal)]);
            end
%             deal with double data and save them
%             noise2=min(min(tempimg2(end-50:end, 2:end)));
%             rawdata=tempimg2-noise;
%             ImName2 = [ Folder2savemat '\' num2str(subBinBaseInd*100, '%4.4d\n') ];
%             save('ImName2.mat','rawdata'); 
            if save_OAC_data == 1
                Folder2save_OAC_mat = [hostfolder '\' FoderName '\' FoderName mask_source_type '_OAC_mat_500frame'];
                if ~exist(Folder2save_OAC_mat)
                        mkdir(Folder2save_OAC_mat);
                end       
                ImName4 = [Folder2save_OAC_mat '\' FoderName '_OAC_' num2str(framenum + framenumCount, '%4.4d\n') '.mat'] ;
                save(ImName4,'u_z2');
            end

            if save_OCT_data == 1
                Folder2save_OCT_mat = [hostfolder '\' FoderName '\' FoderName '_OCT_mat_500frame'];
                if ~exist(Folder2save_OCT_mat)
                        mkdir(Folder2save_OCT_mat);
                end   
                ImName_oct = [Folder2save_OCT_mat '\' FoderName '_OCT_' num2str(framenum + framenumCount, '%4.4d\n') '.mat'] ;
                save(ImName_oct,'tempimg');
            end

            clear tempimg u_z2;
        end       
        framenumCount = framenumCount + frames; 
    end
    clear ZimageDownsize; 
    clear ZimageRescalFlip;
    clear ZimageRescal;
    clear Zimage; 
    clear tempimg;
    clear tempimg3;
    clear tempimg2;
    clear tempa2;
    clear tempa;
    clear u_z2;
    clear intensity2;
    clear intensity3;
    clear Gxx Gyy k2;
    clear a1 a;
    clear framesTotal;
    %% Detect line
    
    if detect_junction == 1
        Folder2saveplot = [hostfolder '\' FoderName '\' FoderName '_plot_500frame'];
        if ~exist(Folder2saveplot)
            mkdir(Folder2saveplot);
        end
        
        Folder2save_epidermis_mat = [hostfolder '\' FoderName '\' FoderName '_epidermis_mat_500frame'];
        if ~exist(Folder2save_epidermis_mat)
            mkdir(Folder2save_epidermis_mat);
        end
        
        Folder2save_epidermis_mask = [hostfolder '\' FoderName '\' FoderName '_epidermis_mask_resize_500frame'];
        if ~exist(Folder2save_epidermis_mask)
            mkdir(Folder2save_epidermis_mask);
        end
        
        Folder2save_epidermis_mask512 = [hostfolder '\' FoderName '\' FoderName '_resize512_epidermis_mask_500frame'];
        if ~exist(Folder2save_epidermis_mask512)
            mkdir(Folder2save_epidermis_mask512);
        end
        
%         Folder2save_dermis_mat = [hostfolder '\' FoderName '\' FoderName '_dermis_mat_500frame'];
%         if ~exist(Folder2save_dermis_mat)
%             mkdir(Folder2save_dermis_mat);
%         end
      
        %% caluate gradient to detect epidermis
        %setting
        %u_z3=cell2mat(u_z3_cell);
        K=zeros(lines,page);
        K_EDJ=zeros(lines,page);
        ep_u=zeros(lines,page);
        dm_u=zeros(lines,page);
        CF=zeros(lines,page);
        O2=zeros(500,lines);
        %start
        for Frame = 1:1:page
            tic;
            O2=OAC_image_cell{Frame,1}(1:500,:);
            [thr,sorh,keepapp] = ddencmp('den','wv',O2);
            ixc=wdencmp('gbl',O2,'sym4',5,thr,sorh,keepapp);
            k3=medfilt2(ixc,[15 15]);
            isuo=imresize(k3,1, 'bicubic' );
            [Gx, Gy] = imgradientxy(isuo,'sobel');
            if background_line>0
                Gy(background_line-3:background_line+3,:)=0;
            end
            clear Gx k3;
            %% detect upper line of epidermis
            X=zeros(1,lines);
            for x=1:1:lines
                [M,N]=max(Gy(10:200,x)); %set the boundary % if sample is close to DC line, the boudary is neccessory to be changed.
                X(1,x)=N+10;  % if sample is close to DC line, the boudary is neccessory to be changed.
            end
            
            for x=6:1:lines-10
                if abs(X(:,x-1)-X(:,x))>10
                    X(:,x)=(X(:,x-3)+X(:,x+3))/2;
                elseif abs(X(:,x-3)-X(:,x))>20
                    X(:,x)=(X(:,x-5)+X(:,x+5))/2;
                else
                    X(:,x)=X(:,x);
                end
            end
            
            for x=1:1:50 %for bug of edges
                if abs(X(:,x)-X(:,x+10))>30
                    X(:,x)=X(:,x+20);
                else
                    X(:,x)=X(:,x);
                end
            end

            Gy2=Gy;
            Y=zeros(1,lines);
            
            for x=1:1:lines


              q=int16(X(1,x));
              [M2,N2]=max(Gy2(q+20:q+48,x));
              Y(1,x)=N2+q+20;
            end
            
            
            for x=6:1:lines-10
                if abs(Y(:,x-1)-Y(:,x))>5
                    Y(:,x)=(Y(:,x-3)+Y(:,x+3))/2;
                elseif abs(Y(:,x-3)-Y(:,x))>15
                    Y(:,x)=(Y(:,x-3)+Y(:,x+10))/2;
                elseif abs(Y(:,x-3)-Y(:,x))>20
                    Y(:,x)=(Y(:,x-5)+Y(:,x+5))/2;
                else
                    Y(:,x)=Y(:,x);
                end
            end
            clear Gy2;


            %% combine
            f=figure('visible','off');hold on;
            %     imagesc(O2(:,80:end-20)),colormap(gray),hold on;
            %     plot(X(1,80:end-20),'r','LineWidth',0.8),hold on;
            %     plot(Y(1,80:end-20),'b','LineWidth',0.8),hold on;
            imagesc(O2(1:481,30:end-27)),colormap(gray),hold on;
            X_plot=X(1,30:end-27);
            Y_plot=Y(1,30:end-27);

            elapsed_time = toc;
            elapsed_time_mask = elapsed_time_mask + elapsed_time;
            disp(['總執行時間(mask)：', num2str(elapsed_time_mask), ' 秒']);
            
            plot(X_plot(1,:),'r','LineWidth',0.8),hold on;
            plot(Y_plot(1,:),'b','LineWidth',0.8),hold on;
            set(gca,'YDir','reverse'),axis([0 944 0 481]),set(gca,'Position',[0,0,1,0.6]),title('detect epidermis'),xlabel('line (pixel)'),ylabel('depth (pixel)');
            ImName= [Folder2saveplot '\' num2str(Frame)];
            saveas(f,ImName,'png');
            clear X_plot Y_plot;
            %% Calculate thickness
            A=abs(X-Y);
            B=sort(A); % calculate overall thickness
            B=B(1,100:end-100);
            [U1,T]=size(B);
            for line5=1:1:T
                TH(line5,Frame)=B(1,line5);
            end
            %% Save .mat for epidermis and dermis
            %save epidermis curve or line(use cireve now)
            ep_begin_position=round(min(X(1,:)));
            dm_begin_position=round(min(Y(1,:)));
            ep_depth=(round(max(Y(1,:)))-round(min(X(1,:))))+1;
            epidermis=zeros(ep_depth,lines);
            epidermis_mask=zeros(a_scan,lines);
            for line3=1:1:lines
                epidermis((round(X(1,line3))-ep_begin_position+1):(round(Y(1,line3))-ep_begin_position+1),line3)=O2(round(X(1,line3)):(round(Y(1,line3))),line3);
                %epidermis(1:(round(Y(1,line3))-round(X(1,line3))+1),line3)=O2(round(X(1,line3)):(round(Y(1,line3))),line3);%穠穩簫簣穢?玲兜勻朝氐?
                epidermis_mask(round(X(1,line3)):(round(Y(1,line3))),line3)=255;
            end
            ImName = [Folder2save_epidermis_mat '\' FoderName '_epidermis_' num2str(Frame) '.mat'] ;
            save(ImName,'epidermis');
            
            epidermis_mask=uint8(epidermis_mask);
            epidermis_mask=epidermis_mask(10:379,30:end-27);  % if sample is close to DC line, the boudary is neccessory to be changed.
            %epidermis_mask2=imresize(epidermis_mask,[128 128]);
            epidermis_mask2=imresize(epidermis_mask,[512 512]);
            ImName_mask = [Folder2save_epidermis_mask '\' FoderName '_' num2str(Frame) '.png'] ;
            ImName_mask2 = [Folder2save_epidermis_mask512 '\' FoderName '_' num2str(Frame) '.png'] ;
            imwrite(epidermis_mask,ImName_mask,'png');
            imwrite(epidermis_mask2,ImName_mask2,'png');
            

             clear epidermis;
             clear epidermis_mask;
             clear dermis;
             clear O2;
            %% atteuation coefficients of epidermis and dermis
            
            for line3=1:1:lines
                ep_u(line3,Frame)=mean(u_z3_cell{Frame,1}(round(X(:,line3)*4):round(Y(:,line3)*4),line3));
                dm_u(line3,Frame)=mean(u_z3_cell{Frame,1}(round(Y(:,line3)*4):(round(Y(:,line3)*4)+80),line3));
                %            ep_u(line3,Frame)=mean(u_z3(round(X(:,line3)*4):round(Y(:,line3)*4),line3,Frame));
                %            dm_u(line3,Frame)=mean(u_z3(round(Y(:,line3)*4):(round(Y(:,line3)*4)+80),line3,Frame));
            end
            %u_z3_cell1{Frame,1}=u_z3_cell{Frame,1}(1:4:end,:); 
            % draw 3D picture
            for line3=1:1:lines
                [P]=A(1,line3);
                K(line3,Frame)=P;
                %EDJ~EDJ+20
                K_EDJ(line3,Frame)=Y(1,line3);
            end
            % fitting curve
            X=X(1,80:end-20);
            Y=Y(1,80:end-20);
            x=1:1:901;
            X=X(1,x);
            p1=polyfit(x,X,fitting_order);
            x1 = linspace(1,901,901);
            y1 = polyval(p1,x1);
            Y=Y(1,x);
            p2=polyfit(x,Y,fitting_order);
            x2 = linspace(1,901,901);
            y2 = polyval(p2,x2);
            % calculate smooth
            upperfit=abs(y1-X);
            C=sort(upperfit);
            C=C(100:end-100);
            junctionfit=abs(y2-Y);
            D=sort(junctionfit);
            D=D(100:end-100);
            [U2,S]=size(C);
            for line6=1:1:S
                [C1]=C(1,line6);
                [D1]=D(1,line6);
                surface(line6,Frame)=C1;
                EDJ(line6,Frame)=D1;
            end
            
        end

        % draw 3D picture
        DX=1:1:page;
        DY=1:1:lines;
        [DXX,DYY]=meshgrid(DX,DY);
        DZ=K(DY,DX);
        clear K;
        %figure(100),mesh(DZ),colorbar,hold on
        DZ=DZ.*pixel_size_um./Refractive_index_epidermis;
        TH=TH.*pixel_size_um./Refractive_index_epidermis;
        overall_mean_thickness=mean(mean(TH));
        SD_thickness=std(std(TH));

        
        % calculate smooth
        clear CF;
        surface=surface.*pixel_size_um./Refractive_index_epidermis;
        EDJ=EDJ*pixel_size_um./Refractive_index_epidermis;
        surface_difference=mean(mean(surface));
        SD_surface=std(std(surface));
        EDJ_difference=mean(mean(EDJ));
        SD_EDJ=std(std(EDJ));
        % average overall of frames
        avg_ep_u=mean(mean(ep_u(80:end-20,:)))*10^(-3);
        std_ep_u=std(std(ep_u(80:end-20,:))*10^(-3));
        avg_dm_u=mean(mean(dm_u(80:end-20,:)))*10^(-3);
        std_dm_u=std(std(dm_u(80:end-20,:))*10^(-3));
        % Save Results Analysis
        filename2= [hostfolder '\' FoderName '\' FoderName  mask_source_type  '_500Results Analysis.txt'];
        fid=fopen(filename2,'wt');
        fprintf(fid,'Correction_500frame\naverage epidermal thickness: %6.3fμm, standard deviation(μm): %6.3fμm\n\nsurface smoothness(fitting order=3)\ndifference average:%6.3fμm, standard deviation(μm): %6.3fμm\n\nEDJ smoothness(fitting order=3)\ndifference average:%6.3fμm, standard deviation(μm): %6.3fμm\n\nparameters of optical attenuation coefficient(μm)\naverage of epidermis:%6.3f (1/mm), standard deviation(μm): %6.3f (1/mm)\naverage of dermis(~20 pixels):%6.3f (1/mm), standard deviation(μm): %6.3f (1/mm)',overall_mean_thickness,SD_thickness,surface_difference,SD_surface,EDJ_difference,SD_EDJ,avg_ep_u,std_ep_u,avg_dm_u,std_dm_u);
        fclose(fid);
    end
 clear a  dm_u DX DXX DY DYY DZ EDJ ep_u  Gy isuo ixc T2 OAC_image_cell TH u_z3_cell upperfit surface X_surface X Y y1 y2 junctionfit u_z2_ratio TH
% close all;
toc
end
close(h);
toc
disp('finish processing')