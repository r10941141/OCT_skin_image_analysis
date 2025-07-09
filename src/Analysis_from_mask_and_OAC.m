hostfolder = '..\data\public\raw_data'; %folder containing all cases
mask_file = '..\data\mask_path'; 
mask_source_type = 'model';
background_line = 0; %need to adjust %background setting
DyRange = 30; % setting dynamic range of the log scale OCT image
pixel_size_um=5.92;
Refractive_index_epidermis=1.424;
lines = 1000;
page = 125;
a_scan = 1024;
fitting_order = 3;


filebase1=dir(hostfolder);

for q=1:1:(length(filebase1(:))-2)

            
        FoderName = filebase1(2+q).name;

        tempimg1=zeros(4096,lines);
        tempimg3=zeros(4096,lines);
           

        OAC_image_cell=cell(page,1);
        O2=zeros(500,lines);
        u_z3_cell=cell(page,1);


        Folder2save_epidermis_mat = [hostfolder '\' FoderName '\' FoderName '_epidermis_mat_500frame_' mask_source_type];
        if ~exist(Folder2save_epidermis_mat)
            mkdir(Folder2save_epidermis_mat);
        end       
        Folder2saveplot = [hostfolder '\' FoderName '\' FoderName '_plot_500frame_' mask_source_type];
        if ~exist(Folder2saveplot)
            mkdir(Folder2saveplot);
        end

        


%% caluate gradient to detect epidermis
        %setting
        %u_z3=cell2mat(u_z3_cell);
        K=zeros(lines,page);
        K_EDJ=zeros(lines,page);
        ep_u=zeros(lines,page);
        dm_u=zeros(lines,page);
        CF=zeros(lines,page);
        %start
        for Frame = 1:1:page

            oac_file = [hostfolder FoderName '\' FoderName '_OAC_mat_500frame\' ];
            oct_file = [hostfolder FoderName '\' FoderName '_OCT_mat_500frame\' ];
            oac_filebase = dir(oac_file);
            oct_filebase = dir(oct_file);
            oac_name = oac_filebase(2 + Frame).name;
            oct_name = oct_filebase(2 + Frame).name;
            oac_path = [oac_file oac_name];
            oct_path = [oct_file oct_name];

            tempimg = load(oct_path);
            u_z3_oac = load(oac_path);
            tempimg1 = tempimg.tempimg;
            u_z3 = u_z3_oac.u_z2;


            mask_num =  num2str(Frame , '%04d');
            maskname = [mask_file '\' FoderName '_' mask_num '.png'];
            mask = imread(maskname);


            
            tempimg3(:,:)=tempimg1(1:4096,:);
            u_z2=zeros(4096,lines);tempimg1;
            u_z2=u_z3;
            u_z2_ratio=u_z2.*1.76;
            tempimg3(u_z2>0)=double(u_z2_ratio(u_z2_ratio>0));   
            u_z3_cell{Frame,1} = u_z2(1:2000,:);

            ZimageOAC= abs(squeeze(10*log10(tempimg3(1:4096, :)))); %OAC
            noise = mean(ZimageOAC(end/2:end,:),'all'); 
            ZimageRescalOAC = round((ZimageOAC-noise).*65536/DyRange);  % assume 55dB dynamic range
            ZimageRescalFlipOAC = ZimageRescalOAC;
            ZimageDownsizeOAC=ZimageRescalFlipOAC(4:4:end,:);
            a1=real(uint16(ZimageDownsizeOAC));
            OAC_image_cell{Frame,1}=a1(:,:);
       
            O2=OAC_image_cell{Frame,1}(1:500,:);


            %        O2=O1(1:500,:,Frame);
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

            % detect line from mask
            T1=zeros(1,944);
            T2=zeros(1,944);
            for x=1:1:944
                has_data = mask(:,x) > 127;
                first_data_pos = find(has_data, 1, 'first');
                T1(1,x)=first_data_pos;
            end
            for x=1:1:944
                has_data = mask(:,x) > 127;
                first_data_pos = find(has_data, 1, 'last');
                T2(1,x)=first_data_pos;
            end
            

            % detect line use gradient
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
            X(1,30:end-27)=T1+10;
            Y(1,30:end-27)=T2+10;
            
            f=figure('visible','off');hold on;

            imagesc(O2(1:481,30:end-27)),colormap(gray),hold on;
            X_plot=X(1,30:end-27);
            Y_plot=Y(1,30:end-27);
            plot(X_plot(1,:),'r','LineWidth',0.8),hold on;
            plot(Y_plot(1,:),'b','LineWidth',0.8),hold on;
            set(gca,'YDir','reverse'),axis([0 944 0 481]),set(gca,'Position',[0,0,1,0.6]),title('detect epidermis'),xlabel('line (pixel)'),ylabel('depth (pixel)');
            ImName= [Folder2saveplot '\' num2str(Frame)];
            saveas(f,ImName,'png');

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
            ImName = [Folder2save_epidermis_mat '\' FoderName '_epidermis_' mask_source_type num2str(Frame) '.mat'] ;
            save(ImName,'epidermis');
            
            epidermis_mask=uint8(epidermis_mask);
            epidermis_mask=epidermis_mask(10:379,30:end-27);  

            

            %% atteuation coefficients of epidermis and dermis
            
            for line3=1:1:lines
                ep_u(line3,Frame)=mean(u_z3_cell{Frame,1}(round(X(:,line3)*4):round(Y(:,line3)*4),line3));
                dm_u(line3,Frame)=mean(u_z3_cell{Frame,1}(round(Y(:,line3)*4):(round(Y(:,line3)*4)+80),line3));

            end
            u_z3_cell{Frame,1}=u_z3_cell{Frame,1}(1:4:end,:); 
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
        filename2= [hostfolder '\' FoderName '\' FoderName '_500Results ' mask_source_type 'mask Analysis.txt'];
        fid=fopen(filename2,'wt');
        fprintf(fid,'Correction_500frame\naverage epidermal thickness: %6.3fμm, standard deviation(μm): %6.3fμm\n\nsurface smoothness(fitting order=3)\ndifference average:%6.3fμm, standard deviation(μm): %6.3fμm\n\nEDJ smoothness(fitting order=3)\ndifference average:%6.3fμm, standard deviation(μm): %6.3fμm\n\nparameters of optical attenuation coefficient(μm)\naverage of epidermis:%6.3f (1/mm), standard deviation(μm): %6.3f (1/mm)\naverage of dermis(~20 pixels):%6.3f (1/mm), standard deviation(μm): %6.3f (1/mm)',overall_mean_thickness,SD_thickness,surface_difference,SD_surface,EDJ_difference,SD_EDJ,avg_ep_u,std_ep_u,avg_dm_u,std_dm_u);
        fclose(fid);
end